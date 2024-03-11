import os, sys
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn

import numpy as np
from skimage.io import imread
# from .utils.renderer import SRenderY, set_rasterizer
from .utils.renderer3 import SRenderY, set_rasterizer
from .models.encoders import ResnetEncoder
from .models.encoders_au import AUEncoder
from .models.encoders_au_v2 import AUEncoder2
from .models.FLAME import FLAME, FLAMETex
from .models.decoders import Generator
from .models.enc_net import Resnet50Encoder, Resnet50Encoder_v2
from .utils import util
from .utils.rotation_converter import batch_euler2axis
from .utils.tensor_cropper import transform_points
from .utils.config import cfg
from .utils.util import au_weights
from .models.OpenGraphAU.model.ANFL import AFG
from .models.GATE import GATE
from .models.encoders_au import AUEncoder
# from .models.TransformerDecoder import FLAMETransformerDecoderHead
from .models.OpenGraphAU.utils import load_state_dict
from .models.OpenGraphAU.utils import *
from .models.OpenGraphAU.conf import get_config,set_logger,set_outdir,set_env
# from .models.gat import GAT

torch.backends.cudnn.benchmark = True
# PerspectiveCameras.transform_points_screen()
class DECA(nn.Module):
    def __init__(self, config=None, device='cuda:1'):
        super(DECA, self).__init__()
        if config is None:
            self.cfg = cfg
        else:
            self.cfg = config
        self.device = device
        self.image_size = self.cfg.dataset.image_size
        self.uv_size = self.cfg.model.uv_size
        self.flame_mask_file = self.cfg.model.flame_mask_path
        self.batchsize = self.cfg.dataset.batch_size
        self.au_weight = au_weights()
        self._create_model(self.cfg.model)
        self._setup_renderer(self.cfg.model)
        self.n_scenelight = self.cfg.model.n_scenelight
        self.n_facelight = self.cfg.model.n_facelight

    def _setup_renderer(self, model_cfg):
        set_rasterizer(self.cfg.rasterizer_type)
        self.render = SRenderY(self.image_size, flame_mask_file=self.flame_mask_file, obj_filename=model_cfg.topology_path, uv_size=model_cfg.uv_size,
        # self.render = SRenderY(self.image_size, obj_filename=model_cfg.topology_path, uv_size=model_cfg.uv_size,
                               rasterizer_type=self.cfg.rasterizer_type).to(self.device)
        # self.render = SRenderY(self.image_size, flame_mask_file=self.flame_mask_file, obj_filename=model_cfg.topology_path, uv_size=model_cfg.uv_size,
                            #    rasterizer_type=self.cfg.rasterizer_type).to(self.device)
        # face mask for rendering details
        mask = imread(model_cfg.face_eye_mask_path).astype(np.float32) / 255.;
        mask = torch.from_numpy(mask[:, :, 0])[None, None, :, :].contiguous()
        self.uv_face_eye_mask = F.interpolate(mask, [model_cfg.uv_size, model_cfg.uv_size]).to(self.device)
        mask = imread(model_cfg.face_mask_path).astype(np.float32) / 255.;
        mask = torch.from_numpy(mask[:, :, 0])[None, None, :, :].contiguous()
        self.uv_face_mask = F.interpolate(mask, [model_cfg.uv_size, model_cfg.uv_size]).to(self.device)
        # displacement correction
        fixed_dis = np.load(model_cfg.fixed_displacement_path)
        self.fixed_uv_dis = torch.tensor(fixed_dis).float().to(self.device)
        # mean texture
        mean_texture = imread(model_cfg.mean_tex_path).astype(np.float32) / 255.;
        mean_texture = torch.from_numpy(mean_texture.transpose(2, 0, 1))[None, :, :, :].contiguous()
        self.mean_texture = F.interpolate(mean_texture, [model_cfg.uv_size, model_cfg.uv_size]).to(self.device)
        # dense mesh template, for save detail mesh
        self.dense_template = np.load(model_cfg.dense_template_path, allow_pickle=True, encoding='latin1').item()

        self.lightprobe_normal_images = F.interpolate(torch.from_numpy(np.load(model_cfg.lightprobe_normal_path)).float(), [model_cfg.image_size, model_cfg.image_size]).to(self.device)
        self.lightprobe_albedo_images = F.interpolate(torch.from_numpy(np.load(model_cfg.lightprobe_albedo_path)).float(), [model_cfg.image_size, model_cfg.image_size]).to(self.device)
        

    def _create_model(self, model_cfg):
        # set up parameters
        self.n_param =  50 + model_cfg.n_exp + model_cfg.n_pose + model_cfg.n_cam + model_cfg.n_light + model_cfg.n_shape
        # self.n_cond = model_cfg.n_exp + 3  # exp + jaw pose
        self.n_detail = model_cfg.n_detail
        self.n_movem = model_cfg.n_exp + 3 # + 27  # exp + jaw pose + action unit
        self.param_dict = {i: model_cfg.get('n_' + i) for i in model_cfg.param_list}
        # self.param_dict['tex'] = 50

        # au config
        self.auconf = get_config()
        self.auconf.evaluate = True
        self.auconf.gpu_ids = os.environ["CUDA_VISIBLE_DEVICES"]
        set_env(self.auconf)

        # encoders
        # self.E_flame = ResnetEncoder(outsize=self.n_param).to(self.device)
        self.E_flame = ResnetEncoder(outsize=self.n_param).to(self.device)
        self.E_detail = ResnetEncoder(outsize=self.n_detail).to(self.device)
        self.E_detail_deca = ResnetEncoder(outsize=self.n_detail).to(self.device)
        self.D_detail_deca = Generator(latent_dim=self.n_detail + self.n_movem, out_channels=1, out_scale=model_cfg.max_z,
                                  sample_mode='bilinear').to(self.device)
        
        self.E_albedo = Resnet50Encoder_v2(outsize=model_cfg.n_tex).to(self.device)
        self.E_scene_light = Resnet50Encoder(outsize=model_cfg.n_scenelight).to(self.device)
        self.E_face_light = Resnet50Encoder(outsize=model_cfg.n_facelight).to(self.device)
        scene_model_path = model_cfg.pretrained_modelpath_scene
        facel_model_path = model_cfg.pretrained_modelpath_facel
        albedo_model_path = model_cfg.pretrained_modelpath_albedo

        self.AUNet = AFG(num_main_classes=self.auconf.num_main_classes, num_sub_classes=self.auconf.num_sub_classes, backbone=self.auconf.arc).to(self.device)
        self.AU_Encoder = AUEncoder().to(self.device)
        self.AUD_Encoder = AUEncoder2().to(self.device)
        self.GATE = GATE(nfeat=512, 
                nhid=512, 
                noutput=59, 
                dropout=0.0, 
                nheads=4, 
                alpha=0.2,
                batchsize=self.batchsize).to(self.device)
        self.GATE_detail = GATE(nfeat=512, 
                nhid=512, 
                noutput=128, 
                dropout=0.0, 
                nheads=1, 
                alpha=0.2,
                batchsize=self.batchsize).to(self.device)
        # self.FTD = FLAMETransformerDecoderHead().to(self.device)

        self.AUNet = load_state_dict(self.AUNet, self.auconf.resume).to(self.device)
        # decoders
        self.flame = FLAME(model_cfg).to(self.device)
        if model_cfg.use_tex:
            self.flametex = FLAMETex(model_cfg).to(self.device)
        self.D_detail = Generator(latent_dim=self.n_detail + self.n_movem, out_channels=1, out_scale=model_cfg.max_z,
                                  sample_mode='bilinear').to(self.device)
        # resume model
        model_path = self.cfg.pretrained_modelpath
        model_path_224 = self.cfg.pretrained_modelpath_224
        if os.path.exists(model_path):
            print(f'trained model found. load {model_path}')
            checkpoint = torch.load(model_path)
            self.checkpoint = checkpoint
            # print(checkpoint.keys())
            # util.copy_state_dict(self.E_flame.state_dict(), checkpoint['E_flame'])
            util.copy_state_dict(self.AU_Encoder.state_dict(), checkpoint['AU_Encoder'])
            # if 'GATE' in checkpoint.keys():
            util.copy_state_dict(self.GATE.state_dict(), checkpoint['GATE'])
            util.copy_state_dict(self.E_detail.state_dict(), checkpoint['E_detail'])
            util.copy_state_dict(self.D_detail.state_dict(), checkpoint['D_detail'])
            # util.copy_state_dict(self.GATE_detail.state_dict(), checkpoint['GATE_detail'])
                # util.copy_state_dict(self.GATE.state_dict(), checkpoint['GAT'])

            # if 'E_detail' in checkpoint.keys():
            #     util.copy_state_dict(self.E_detail.state_dict(), checkpoint['E_detail'])
            # if 'D_detail' in checkpoint.keys():
            #     util.copy_state_dict(self.D_detail.state_dict(), checkpoint['D_detail'])
            # util.copy_state_dict(self.FTD.state_dict(), checkpoint['FTD'])
        # else:
        #     print(f'please check model path: {model_path}')
            # exit()
        if os.path.exists(model_path_224):
            print(f'trained model found. load {model_path_224}')
            checkpoint = torch.load(model_path_224)
            util.copy_state_dict(self.E_flame.state_dict(), checkpoint['E_flame'])
            util.copy_state_dict(self.E_detail_deca.state_dict(), checkpoint['E_detail'])
            util.copy_state_dict(self.D_detail_deca.state_dict(), checkpoint['D_detail'])
            if os.path.exists(scene_model_path):
                print('trained model found. load {}'.format(scene_model_path))
                checkpoint = torch.load(scene_model_path)
                util.copy_state_dict(self.E_scene_light.state_dict(), checkpoint['E_scene_light'])
            if os.path.exists(facel_model_path):
                print('trained model found. load {}'.format(facel_model_path))
                checkpoint = torch.load(facel_model_path)
                util.copy_state_dict(self.E_face_light.state_dict(), checkpoint['E_face_light'])
            if os.path.exists(albedo_model_path):
                print('trained model found. load {}'.format(albedo_model_path))
                checkpoint = torch.load(albedo_model_path)
                util.copy_state_dict(self.E_albedo.state_dict(), checkpoint['E_albedo'])
        else:
            print(f'please check pretrained deca model path: {model_path_224}')
            exit()
        # eval mode

        # self.E_flame.eval()
        self.E_flame.eval()
        self.E_detail.eval()
        self.D_detail.eval()
        self.E_scene_light.eval()
        self.E_face_light.eval()
        self.E_albedo.eval()
        self.AUNet.eval()
        self.GATE.eval()
        self.GATE_detail.eval()
        self.AUD_Encoder.eval()
        self.AU_Encoder.eval()

    def decompose_code(self, code, num_dict):
        '''
        config.n_shape + config.n_tex + config.n_exp + config.n_pose + config.n_cam + config.n_light
        '''
        # code_list = []
        # num_list = [self.cfg.model.n_shape, 50, self.cfg.model.n_exp, self.cfg.model.n_pose, self.cfg.model.n_cam, self.cfg.model.n_light]
        # start = 0
        # for i in range(len(num_list)):
        #     code_list.append(code[:, start:start+num_list[i]])
        #     start = start + num_list[i]
        # code_list[-1] = code_list[-1].reshape(code.shape[0], 9, 3)
        # return code_list
        ''' Convert a flattened parameter vector to a dictionary of parameters
        code_dict.keys() = ['shape', 'tex', 'exp', 'pose', 'cam', 'light']
        '''
        code_dict = {}
        # num_list = [self.cfg.model.n_shape, 50, self.cfg.model.n_exp, self.cfg.model.n_pose, self.cfg.model.n_cam, self.cfg.model.n_light]
        start = 0
        for key in num_dict:
            if key != 'tex':
                end = start + int(num_dict[key])
            else:
                end = start + 50
            code_dict[key] = code[:, start:end]
            start = end
            if key == 'light':
                code_dict[key] = code_dict[key].reshape(code_dict[key].shape[0], 9, 3)
                
        return code_dict

    def displacement2normal(self, uv_z, coarse_verts, coarse_normals):
        ''' Convert displacement map into detail normal map
        '''
        batch_size = uv_z.shape[0]
        uv_coarse_vertices = self.render.world2uv(coarse_verts).detach()
        uv_coarse_normals = self.render.world2uv(coarse_normals).detach()

        uv_z = uv_z * self.uv_face_eye_mask
        uv_detail_vertices = uv_coarse_vertices + uv_z * uv_coarse_normals + self.fixed_uv_dis[None, None, :,
                                                                             :] * uv_coarse_normals.detach()
        dense_vertices = uv_detail_vertices.permute(0, 2, 3, 1).reshape([batch_size, -1, 3])
        uv_detail_normals = util.vertex_normals(dense_vertices, self.render.dense_faces.expand(batch_size, -1, -1))
        uv_detail_normals = uv_detail_normals.reshape(
            [batch_size, uv_coarse_vertices.shape[2], uv_coarse_vertices.shape[3], 3]).permute(0, 3, 1, 2)
        uv_detail_normals = uv_detail_normals * self.uv_face_eye_mask + uv_coarse_normals * (1. - self.uv_face_eye_mask)
        return uv_detail_normals

    def visofp(self, normals):
        ''' visibility of keypoints, based on the normal direction
        '''
        normals68 = self.flame.seletec_3d68(normals)
        vis68 = (normals68[:, :, 2:] < 0.1).float()
        return vis68

    # @torch.no_grad()
    def encode(self, images_224,  scene_image=None, use_detail=True, use_gnn=False):
        # if use_detail:
        #     # use_detail is for training detail model, need to set coarse model as eval mode
        #     with torch.no_grad():
        #         parameters_224 = self.E_flame_224(images_224)
        # else:
            # t1 = time.time()
        B, C, H, W = images_224.size()
        if scene_image is None:
            scene_image = images_224

        if use_detail or use_gnn:
            x, afn, main_cl = self.AUNet(images_224, use_gnn=True)
        else:
            x, afn = self.AUNet(images_224, use_gnn=use_gnn)

        # full_rep = x.mean(dim=-2).view([self.batchsize, 1, -1])
        # feats = torch.concat([main_sc, full_rep], dim=1)
        x = self.E_flame.forwardResNet(images_224)
        full_rep = self.AU_Encoder(x)
        full_rep = full_rep.unsqueeze(1)
        feats = torch.concat([afn, full_rep], dim=1)
        parameters = self.GATE(feats)
        # parameters = self.FTD(hprime)
        # parameters = self.AU_Encoder(x, hprime)
        with torch.no_grad():
            parameters_224 = self.E_flame(images_224)
            # print("Encode time..", (time.time() - t1)*1000)
        codedict = self.decompose_code(parameters_224, self.param_dict)

        with torch.no_grad():
            codedict['S_light'] = self.E_scene_light(images_224)
            codedict['F_light'] = self.E_face_light(images_224).reshape(B, 9, 3)

        # codedict['shape'] = parameters[:,:100]
        codedict['exp'] = parameters[:,:50]
        codedict['pose'][:,3:] = parameters[:,50:53]

        codedict['pose'][:,:3] = codedict['pose'][:,:3] + parameters[:,53:56]
        codedict['cam'] = codedict['cam'][:,:3] + parameters[:,56:59]

        # codedict['cam'] = parameters[:,156:159]
        codedict['images'] = images_224
        # # 'tex', 'exp', 'pose', 'cam', 'light'noutput
        if use_detail or use_gnn:
            codedict['au'] = main_cl
        if use_detail:
            if self.cfg.model.use_tex and self.cfg.model.tex_type == 'BalanceAlb':
                lightcode, scale_factors, normalized_sh_params = self.fuse_light(codedict['S_light'], codedict['F_light'])
                codedict['light'] = lightcode
                # print(codedict['light'])
                images_224 = images_224.contiguous().view(B, C, H, W)
                scale_factors_img = scale_factors.contiguous().view(B, scale_factors.size(1), 1, 1).repeat(1, 1, H, W)
                images_cond = torch.cat((scale_factors_img, images_224), dim=1)
                texcode = self.E_albedo(images_cond)
                codedict['tex'] = texcode
                albedo = self.flametex(texcode)
            
            # x = self.E_detail.forwardResNet(images_224)
            # full_rep = self.AUD_Encoder(x)
            # full_rep = full_rep.unsqueeze(1)
            # feats = torch.concat([full_rep, afn], dim=1)
            # detailcode = self.GATE_detail(feats)
            detailcode = self.E_detail_deca(images_224)
            codedict['detail'] = detailcode
        if self.cfg.model.jaw_type == 'euler':
            posecode = codedict['pose']
            euler_jaw_pose = posecode[:, 3:].clone()  # x for yaw (open mouth), y for pitch (left ang right), z for roll
            posecode[:, 3:] = batch_euler2axis(euler_jaw_pose)
            codedict['pose'] = posecode
            codedict['euler_jaw_pose'] = euler_jaw_pose
        return codedict
    
    
    # @torch.no_grad()
    def decode(self, codedict, rendering=True, iddict=None, vis_lmk=True, vis_au=False, return_vis=True, use_detail=True, use_gnn=False,
               render_orig=False, original_image=None, tform=None):
        images = codedict['images']
        B, C, H, W = images.size()
        batch_size = images.shape[0]

        ## decode
        verts, landmarks2d, landmarks3d, mp_landmark = self.flame(shape_params=codedict['shape'], expression_params=codedict['exp'],
                                                     pose_params=codedict['pose'])
        # albedo = self.flametex(codedict['tex'])

        if self.cfg.model.use_tex and self.cfg.model.tex_type == 'BalanceAlb':
            lightcode, scale_factors, normalized_sh_params = self.fuse_light(codedict['S_light'], codedict['F_light'])
            codedict['light'] = lightcode
            # print(codedict['light'])
            images = images.contiguous().view(B, C, H, W)
            scale_factors_img = scale_factors.contiguous().view(B, scale_factors.size(1), 1, 1).repeat(1, 1, H, W)
            images_cond = torch.cat((scale_factors_img, images), dim=1)
            texcode = self.E_albedo(images_cond)
            codedict['tex'] = texcode
            albedo = self.flametex(texcode)
            
        elif self.cfg.model.use_tex:
            albedo = self.flametex(codedict['tex'])
        else:   
            albedo = torch.zeros([batch_size, 3, self.uv_size, self.uv_size], device=images.device)
        landmarks3d_world = landmarks3d.clone()

        ## projection
        landmarks2d = util.batch_orth_proj(landmarks2d, codedict['cam'])[:, :, :2];
        landmarks2d[:, :, 1:] = -landmarks2d[:, :, 1:]  # ; landmarks2d = landmarks2d*self.image_size/2 + self.image_size/2
        landmarks3d = util.batch_orth_proj(landmarks3d, codedict['cam']);
        landmarks3d[:, :, 1:] = -landmarks3d[:, :, 1:]  # ; landmarks3d = landmarks3d*self.image_size/2 + self.image_size/2
        mp_landmark = util.batch_orth_proj(mp_landmark, codedict['cam'])[:, :, :2];

        mp_landmark[:, :, 1:] = -mp_landmark[:, :, 1:]

        trans_verts = util.batch_orth_proj(verts, codedict['cam']);
        trans_verts[:, :, 1:] = -trans_verts[:, :, 1:]
        opdict = {
            'verts': verts,
            'trans_verts': trans_verts,
            'landmarks2d': landmarks2d,
            'landmarks3d': landmarks3d,
            'mp_landmark': mp_landmark,
            'landmarks3d_world': landmarks3d_world,
        }
        

        ## rendering
        if return_vis and render_orig and original_image is not None and tform is not None:
            points_scale = [self.image_size, self.image_size]
            _, _, h, w = original_image.shape
            # import ipdb; ipdb.set_trace()
            trans_verts = transform_points(trans_verts, tform, points_scale, [h, w])
            landmarks2d = transform_points(landmarks2d, tform, points_scale, [h, w])

            landmarks3d = transform_points(landmarks3d, tform, points_scale, [h, w])
            mp_landmark = transform_points(mp_landmark, tform, points_scale, [h, w])
            background = original_image
            images = original_image
        else:
            h, w = self.image_size, self.image_size
            # background = None
            background = images

        if rendering:
            # ops = self.render(verts, trans_verts, albedo, codedict['light'])
            ops = self.render(verts, trans_verts, albedo, codedict['light'], h=h, w=w)#, background=background)
            ## output
            opdict['grid'] = ops['grid']
            opdict['rendered_images'] = ops['images']
            opdict['alpha_images'] = ops['alpha_images']
            opdict['normal_images'] = ops['normal_images']
            opdict['v_colors'] = ops['v_colors']

            predicted_images_alpha = ops['images'] * ops['alpha_images']
            predicted_albedo_images = ops['albedo_images'] * ops['alpha_images']
            predicted_shading = self.lightprobe_shading(self.SH_convert(lightcode))

        if self.cfg.model.use_tex:
            opdict['albedo'] = albedo        

        if use_detail:
            uv_z = self.D_detail_deca(torch.cat([codedict['pose'][:, 3:], codedict['exp'], codedict['detail']], dim=1))
            # uv_z = self.D_detail(torch.cat([codedict['pose'][:, 3:], codedict['exp'], codedict['detail'], codedict['au']], dim=1))
            if iddict is not None:
                uv_z = self.D_detail_deca(torch.cat([codedict['pose'][:, 3:], codedict['exp'], codedict['detail']], dim=1))
                # uv_z = self.D_detail(torch.cat([iddict['pose'][:, 3:], iddict['exp'], codedict['detail'], codedict['au']], dim=1))
            uv_detail_normals = self.displacement2normal(uv_z, verts, ops['normals'])
            uv_shading = self.render.add_SHlight(uv_detail_normals, codedict['light'])
            uv_texture = albedo * uv_shading

            opdict['uv_texture'] = uv_texture
            opdict['normals'] = ops['normals']
            opdict['uv_detail_normals'] = uv_detail_normals
            opdict['displacement_map'] = uv_z + self.fixed_uv_dis[None, None, :, :]
        
        if vis_lmk:
            landmarks3d_vis = self.visofp(ops['transformed_normals'])  # /self.image_size
            landmarks3d = torch.cat([landmarks3d, landmarks3d_vis], dim=2)
            opdict['landmarks3d'] = landmarks3d

        if return_vis:
            ## render shape
            # shape_images = self.render.render_shape(verts, trans_verts)
            shape_images_full, shape_images, _, grid, alpha_images = self.render.render_shape(verts, trans_verts, h=h, w=w,
                                                                           images=background, return_grid=True)
            # shape_images, _, grid, alpha_images = self.render.render_shape(verts, trans_verts, h=h, w=w,
                                                                        #    images=background, return_grid=True)
            if use_detail:
                predicted_detail_images = F.grid_sample(uv_texture, ops['grid'].detach(), align_corners=False)

                detail_normal_images = F.grid_sample(uv_detail_normals, grid, align_corners=False) * alpha_images
                shape_detail_images_full, shape_detail_images = self.render.render_shape(verts, trans_verts,
                                                            detail_normal_images=detail_normal_images, h=h, w=w,
                                                            images=background)

            ## extract texture
            ## TODO: current resolution 256x256, support higher resolution, and add visibility
            uv_pverts = self.render.world2uv(trans_verts)
            uv_gt = F.grid_sample(images, uv_pverts.permute(0, 2, 3, 1)[:, :, :, :2], mode='bilinear',
                                  align_corners=False)
            if self.cfg.model.use_tex:
                ## TODO: poisson blending should give better-looking results
                if self.cfg.model.extract_tex:
                    uv_texture_gt = uv_gt[:, :3, :, :] * self.uv_face_eye_mask + (
                                uv_texture[:, :3, :, :] * (1 - self.uv_face_eye_mask))
                else:
                    # uv_texture_gt = uv_texture[:, :3, :, :]
                    uv_texture_gt = albedo[:, :3, :, :]
            else:
                uv_texture_gt = uv_gt[:, :3, :, :] * self.uv_face_eye_mask + (
                            torch.ones_like(uv_gt[:, :3, :, :]) * (1 - self.uv_face_eye_mask) * 0.7)

            opdict['uv_texture_gt'] = uv_texture_gt
            if use_detail:
                visdict = {
                    'inputs': images,
                    # 'predicted_detail_images': predicted_detail_images,
                    # 'predicted_images_alpha': predicted_images_alpha,
                    # 'landmarks2d': util.tensor_vis_landmarks(images, landmarks2d),
                    # 'mp_landmark': util.tensor_vis_landmarks(images, mp_landmark),
                    # 'landmarks3d': util.tensor_vis_landmarks(images, landmarks3d),
                    # 'shape_images': shape_images,
                    # 'render_images2': opdict['rendered_images'],
                    'shape_detail_images': shape_detail_images,
                    'shape_images': shape_images,
                    'render_images': images*(1-opdict['v_colors'])+predicted_detail_images*opdict['v_colors'],#opdict['rendered_images'],
                }
            else:

                visdict = {
                    'inputs': images,
                    # 'predicted_images_alpha': predicted_images_alpha,
                    # 'predicted_albedo_images': predicted_albedo_images,
                    # 'predicted_shading': predicted_shading,
                    # 'landmarks2d': util.tensor_vis_landmarks(images, landmarks2d),
                    # 'mp_landmark': util.tensor_vis_landmarks(images, mp_landmark),
                    # 'au_gt': util.draw_activation_circles(images, landmarks3d, codedict['au'], self.au_weight),
                    # 'au_pred': util.draw_activation_circles(opdict['rendered_images'], opdict['landmarks3d'], opdict['au_rend'], self.au_weight),
                    # 'landmarks3d': util.tensor_vis_landmarks(images, landmarks3d),
                    'shape_images': shape_images,
                    'shape_images_full': shape_images_full,
                    'render_images': images*(1-opdict['v_colors'])+opdict['rendered_images']*opdict['v_colors'],#opdict['rendered_images'],
                    # 'render_images2': opdict['rendered_images'],
                    # 'shape_detail_images': shape_detail_images
                }
            # if self.cfg.model.use_tex:
            #     visdict['uv_texture_gt'] = uv_texture_gt #!!!
                # visdict['rendered_images'] = ops['images']

            return opdict, visdict

        else:
            return opdict


    # @torch.no_grad()
    def encode_deca(self, images_224,  use_detail=True):
        if use_detail:
            # use_detail is for training detail model, need to set coarse model as eval mode
            with torch.no_grad():
                # parameters = self.E_flame(images)
                parameters_224 = self.E_flame(images_224)
        else:
            # t1 = time.time()
            # parameters = self.E_flame(images)
            with torch.no_grad():
                parameters_224 = self.E_flame(images_224)
            # print("Encode time..", (time.time() - t1)*1000)
        codedict = self.decompose_code(parameters_224, self.param_dict)
        codedict['images'] = images_224
        # # 'tex', 'exp', 'pose', 'cam', 'light'
        codedict['shape'] =  parameters_224[:, :100]

        if use_detail:
            detailcode = self.E_detail_deca(images_224)
            codedict['detail'] = detailcode
        if self.cfg.model.jaw_type == 'euler':
            posecode = codedict['pose']
            euler_jaw_pose = posecode[:, 3:].clone()  # x for yaw (open mouth), y for pitch (left ang right), z for roll
            posecode[:, 3:] = batch_euler2axis(euler_jaw_pose)
            codedict['pose'] = posecode
            codedict['euler_jaw_pose'] = euler_jaw_pose
        return codedict


    # @torch.no_grad()
    def decode_deca(self, codedict, rendering=True, iddict=None, vis_lmk=True, vis_au=False, return_vis=True, use_detail=True, use_gnn=False,
               render_orig=False, original_image=None, tform=None):
        images = codedict['images']
        B, C, H, W = images.size()
        batch_size = images.shape[0]

        ## decode
        verts, landmarks2d, landmarks3d, mp_landmark = self.flame(shape_params=codedict['shape'], expression_params=codedict['exp'],
                                                     pose_params=codedict['pose'])
        # albedo = self.flametex(codedict['tex'])

        if self.cfg.model.use_tex and self.cfg.model.tex_type == 'BalanceAlb':
            lightcode, scale_factors, normalized_sh_params = self.fuse_light(codedict['S_light'], codedict['F_light'])
            codedict['light'] = lightcode
            # print(codedict['light'])
            images = images.contiguous().view(B, C, H, W)
            scale_factors_img = scale_factors.contiguous().view(B, scale_factors.size(1), 1, 1).repeat(1, 1, H, W)
            images_cond = torch.cat((scale_factors_img, images), dim=1)
            texcode = self.E_albedo(images_cond)
            codedict['tex'] = texcode
            albedo = self.flametex(texcode)
            
        elif self.cfg.model.use_tex:
            albedo = self.flametex(codedict['tex'])
        else:   
            albedo = torch.zeros([batch_size, 3, self.uv_size, self.uv_size], device=images.device)
        landmarks3d_world = landmarks3d.clone()

        ## projection
        landmarks2d = util.batch_orth_proj(landmarks2d, codedict['cam'])[:, :, :2];
        landmarks2d[:, :, 1:] = -landmarks2d[:, :, 1:]  # ; landmarks2d = landmarks2d*self.image_size/2 + self.image_size/2
        landmarks3d = util.batch_orth_proj(landmarks3d, codedict['cam']);
        landmarks3d[:, :, 1:] = -landmarks3d[:, :, 1:]  # ; landmarks3d = landmarks3d*self.image_size/2 + self.image_size/2
        mp_landmark = util.batch_orth_proj(mp_landmark, codedict['cam'])[:, :, :2];

        mp_landmark[:, :, 1:] = -mp_landmark[:, :, 1:]

        trans_verts = util.batch_orth_proj(verts, codedict['cam']);
        trans_verts[:, :, 1:] = -trans_verts[:, :, 1:]
        opdict = {
            'verts': verts,
            'trans_verts': trans_verts,
            'landmarks2d': landmarks2d,
            'landmarks3d': landmarks3d,
            'mp_landmark': mp_landmark,
            'landmarks3d_world': landmarks3d_world,
        }
        

        ## rendering
        if return_vis and render_orig and original_image is not None and tform is not None:
            points_scale = [self.image_size, self.image_size]
            _, _, h, w = original_image.shape
            # import ipdb; ipdb.set_trace()
            trans_verts = transform_points(trans_verts, tform, points_scale, [h, w])
            landmarks2d = transform_points(landmarks2d, tform, points_scale, [h, w])

            landmarks3d = transform_points(landmarks3d, tform, points_scale, [h, w])
            mp_landmark = transform_points(mp_landmark, tform, points_scale, [h, w])
            background = original_image
            images = original_image
        else:
            h, w = self.image_size, self.image_size
            # background = None
            background = images

        if rendering:
            # ops = self.render(verts, trans_verts, albedo, codedict['light'])
            ops = self.render(verts, trans_verts, albedo, codedict['light'], h=h, w=w)#, background=background)
            ## output
            opdict['grid'] = ops['grid']
            opdict['rendered_images'] = ops['images']
            opdict['alpha_images'] = ops['alpha_images']
            opdict['normal_images'] = ops['normal_images']
            opdict['v_colors'] = ops['v_colors']

            predicted_images_alpha = ops['images'] * ops['alpha_images']
            predicted_albedo_images = ops['albedo_images'] * ops['alpha_images']
            predicted_shading = self.lightprobe_shading(self.SH_convert(lightcode))

        if self.cfg.model.use_tex:
            opdict['albedo'] = albedo        

        if use_detail:
            uv_z = self.D_detail_deca(torch.cat([codedict['pose'][:, 3:], codedict['exp'], codedict['detail']], dim=1))
            # uv_z = self.D_detail(torch.cat([codedict['pose'][:, 3:], codedict['exp'], codedict['detail'], codedict['au']], dim=1))
            if iddict is not None:
                uv_z = self.D_detail(torch.cat([codedict['pose'][:, 3:], codedict['exp'], codedict['detail']], dim=1))
                # uv_z = self.D_detail(torch.cat([iddict['pose'][:, 3:], iddict['exp'], codedict['detail'], codedict['au']], dim=1))
            uv_detail_normals = self.displacement2normal(uv_z, verts, ops['normals'])
            uv_shading = self.render.add_SHlight(uv_detail_normals, codedict['light'])
            uv_texture = albedo * uv_shading

            opdict['uv_texture'] = uv_texture
            opdict['normals'] = ops['normals']
            opdict['uv_detail_normals'] = uv_detail_normals
            opdict['displacement_map'] = uv_z + self.fixed_uv_dis[None, None, :, :]
        
        if vis_lmk:
            landmarks3d_vis = self.visofp(ops['transformed_normals'])  # /self.image_size
            landmarks3d = torch.cat([landmarks3d, landmarks3d_vis], dim=2)
            opdict['landmarks3d'] = landmarks3d

        if return_vis:
            ## render shape
            # shape_images = self.render.render_shape(verts, trans_verts)
            shape_images_full, shape_images, _, grid, alpha_images = self.render.render_shape(verts, trans_verts, h=h, w=w,
                                                                           images=background, return_grid=True)
            # shape_images, _, grid, alpha_images = self.render.render_shape(verts, trans_verts, h=h, w=w,
                                                                        #    images=background, return_grid=True)
            if use_detail:
                predicted_detail_images = F.grid_sample(uv_texture, ops['grid'].detach(), align_corners=False)

                detail_normal_images = F.grid_sample(uv_detail_normals, grid, align_corners=False) * alpha_images
                shape_detail_images_full, shape_detail_images = self.render.render_shape(verts, trans_verts,
                                                            detail_normal_images=detail_normal_images, h=h, w=w,
                                                            images=background)

            ## extract texture
            ## TODO: current resolution 256x256, support higher resolution, and add visibility
            uv_pverts = self.render.world2uv(trans_verts)
            uv_gt = F.grid_sample(images, uv_pverts.permute(0, 2, 3, 1)[:, :, :, :2], mode='bilinear',
                                  align_corners=False)
            if self.cfg.model.use_tex:
                ## TODO: poisson blending should give better-looking results
                if self.cfg.model.extract_tex:
                    uv_texture_gt = uv_gt[:, :3, :, :] * self.uv_face_eye_mask + (
                                uv_texture[:, :3, :, :] * (1 - self.uv_face_eye_mask))
                else:
                    # uv_texture_gt = uv_texture[:, :3, :, :]
                    uv_texture_gt = albedo[:, :3, :, :]
            else:
                uv_texture_gt = uv_gt[:, :3, :, :] * self.uv_face_eye_mask + (
                            torch.ones_like(uv_gt[:, :3, :, :]) * (1 - self.uv_face_eye_mask) * 0.7)

            opdict['uv_texture_gt'] = uv_texture_gt
            if use_detail:
                visdict = {
                    'inputs': images,
                    # 'predicted_detail_images': predicted_detail_images,
                    # 'predicted_images_alpha': predicted_images_alpha,
                    # 'landmarks2d': util.tensor_vis_landmarks(images, landmarks2d),
                    # 'mp_landmark': util.tensor_vis_landmarks(images, mp_landmark),
                    # 'landmarks3d': util.tensor_vis_landmarks(images, landmarks3d),
                    # 'shape_images': shape_images,
                    # 'render_images2': opdict['rendered_images'],
                    'shape_detail_images': shape_detail_images,
                    'shape_images': shape_images,
                    'render_images': images*(1-opdict['v_colors'])+predicted_detail_images*opdict['v_colors'],#opdict['rendered_images'],
                }
            else:

                visdict = {
                    'inputs': images,
                    # 'predicted_images_alpha': predicted_images_alpha,
                    # 'predicted_albedo_images': predicted_albedo_images,
                    # 'predicted_shading': predicted_shading,
                    # 'landmarks2d': util.tensor_vis_landmarks(images, landmarks2d),
                    # 'mp_landmark': util.tensor_vis_landmarks(images, mp_landmark),
                    # 'au_gt': util.draw_activation_circles(images, landmarks3d, codedict['au'], self.au_weight),
                    # 'au_pred': util.draw_activation_circles(opdict['rendered_images'], opdict['landmarks3d'], opdict['au_rend'], self.au_weight),
                    # 'landmarks3d': util.tensor_vis_landmarks(images, landmarks3d),
                    'shape_images': shape_images,
                    'shape_images_full': shape_images_full,
                    'render_images': images*(1-opdict['v_colors'])+opdict['rendered_images']*opdict['v_colors'],#opdict['rendered_images'],
                    # 'render_images2': opdict['rendered_images'],
                    # 'shape_detail_images': shape_detail_images
                }
            # if self.cfg.model.use_tex:
            #     visdict['uv_texture_gt'] = uv_texture_gt #!!!
                # visdict['rendered_images'] = ops['images']

            return opdict, visdict

        else:
            return opdict

    def visualize(self, visdict, size=224, dim=2):
        '''
        image range should be [0,1]
        dim: 2 for horizontal. 1 for vertical
        '''
        assert dim == 1 or dim == 2
        grids = {}
        for key in visdict:
            _, _, h, w = visdict[key].shape
            if dim == 2:
                new_h = size;
                new_w = int(w * size / h)
            elif dim == 1:
                new_h = int(h * size / w);
                new_w = size
            grids[key] = torchvision.utils.make_grid(F.interpolate(visdict[key], [new_h, new_w]).detach().cpu())

        grid = torch.cat(list(grids.values()), dim)
        grid_image = (grid.numpy().transpose(1, 2, 0).copy() * 255)[:, :, [2, 1, 0]]
        grid_image = np.minimum(np.maximum(grid_image, 0), 255).astype(np.uint8)

        return grid_image

    
    def SH_normalization(self, sh_param):
        norm_sh_param = F.normalize(sh_param, p=1, dim=1)
        return norm_sh_param

    def lightprobe_shading(self, sh):
        bz = sh.shape[0]
        lightprobe_normal = self.lightprobe_normal_images.expand(bz, -1, -1, -1)
        shading_image = self.render.add_SHlight(lightprobe_normal, sh)
        return shading_image
        
    def SH_convert(self, sh):
        '''
        rotate SH with pi around x axis
        '''
        gt_sh_inverted = sh.clone()
        gt_sh_inverted[:, 1, :] *= -1
        gt_sh_inverted[:, 2, :] *= -1
        gt_sh_inverted[:, 4, :] *= -1
        gt_sh_inverted[:, 7, :] *= -1

        return gt_sh_inverted

    def fuse_light(self, E_scene_light_pred, E_face_light_pred):
        normalized_sh_params = self.SH_normalization(E_face_light_pred)
        lightcode = E_scene_light_pred.unsqueeze(1).expand(-1, 9, -1) * normalized_sh_params

        return lightcode, E_scene_light_pred, normalized_sh_params

    # def save_obj(self, filename, opdict):
    #     '''
    #     vertices: [nv, 3], tensor
    #     texture: [3, h, w], tensor
    #     '''
    #     i = 0
    #     vertices = opdict['verts'][i].cpu().numpy()
    #     faces = self.render.faces[0].cpu().numpy()
    #     # texture = util.tensor2image(opdict['uv_texture_gt'][i])
    #     uvcoords = self.render.raw_uvcoords[0].cpu().numpy()
    #     uvfaces = self.render.uvfaces[0].cpu().numpy()
    #     # save coarse mesh, with texture and normal map
    #     normal_map = util.tensor2image(opdict['uv_detail_normals'][i] * 0.5 + 0.5)
    #     # print(opdict['normals'].shape)
    #     # normal_map = util.tensor2image(opdict['normals'][i] * 0.5 + 0.5)
    #     # normal_map = None
    #     util.write_obj(filename, vertices, faces,
    #                    # texture=texture,
    #                    uvcoords=uvcoords,
    #                    uvfaces=uvfaces,
    #                    normal_map=normal_map)from .utils.util import au_weights
    #     #                dense_faces,
    #     #                colors=dense_colors,
    #     #                inverse_face_order=True)
    def save_obj(self, filename, opdict):
        '''
        vertices: [nv, 3], tensor
        texture: [3, h, w], tensor
        '''
        i = 0
        vertices = opdict['verts'][i].cpu().numpy()
        faces = self.render.faces[0].cpu().numpy()
        texture = util.tensor2image(opdict['uv_texture_gt'][i])
        uvcoords = self.render.raw_uvcoords[0].cpu().numpy()
        uvfaces = self.render.uvfaces[0].cpu().numpy()
        # save coarse mesh, with texture and normal map
        normal_map = util.tensor2image(opdict['uv_detail_normals'][i] * 0.5 + 0.5)
        # normal_map = util.tensor2image(opdict['normals'][i] * 0.5 + 0.5)
        # normal_map = None
        util.write_obj(filename, vertices, faces,
                       texture=texture,
                       uvcoords=uvcoords,
                       uvfaces=uvfaces,
                       normal_map=normal_map)
        # upsample mesh, save detailed mesh
        # texture = texture[:, :, [2, 1, 0]]
        # normals = opdict['normals'][i].cpu().numpy()
        # displacement_map = opdict['displacement_map'][i].cpu().numpy().squeeze()
        # dense_vertices, dense_colors, dense_faces = util.upsample_mesh(vertices, normals, faces, displacement_map,
        #                                                                texture, self.dense_template)
        # util.write_obj(filename.replace('.obj', '_detail.obj'),
        #                dense_vertices,
        #                dense_faces,
        #                colors=dense_colors,
        #                inverse_face_order=True)

    # def run(self, imagepath, iscrop=True):
    #     ''' An api for running deca given an image path
    #     '''
    #     testdata = datasets.TestData(imagepath)
    #     images = testdata[0]['image'].to(self.device)[None, ...]
    #     images_224 = testdata[0]['image_224'].to(self.device)[None, ...]
    #     codedict = self.encode(images, images_224)
    #     opdict, visdict = self.decode(codedict)
    #     return codedict, opdict, visdict

    def model_dict(self):
        return {
            'GATE': self.GATE.state_dict(),
            # 'FTD': self.FTD.state_dict(),
            'AU_Encoder': self.AU_Encoder.state_dict(),
            'GATE_detail': self.GATE_detail.state_dict(),
            'AUD_Encoder': self.AUD_Encoder.state_dict(),
            # 'AU_Net': self.AUNet.state_dict(),
            # 'E_flame': self.E_flame.state_dict(),
            'E_detail': self.E_detail.state_dict(),
            'D_detail': self.D_detail.state_dict()
        }
