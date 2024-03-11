import os, sys
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn

import numpy as np
from skimage.io import imread
from .utils.renderer2 import SRenderY, set_rasterizer
from .models.encoders import ResnetEncoder
from .models.FLAME_deca import FLAME, FLAMETex
from .models.decoders import Generator
from .utils import util
from .utils.rotation_converter import batch_euler2axis
from .utils.tensor_cropper import transform_points
from .utils.config import cfg


torch.backends.cudnn.benchmark = True
# PerspectiveCameras.transform_points_screen()
class DECA(nn.Module):
    def __init__(self, config=None, device='cuda'):
        super(DECA, self).__init__()
        if config is None:
            self.cfg = cfg
        else:
            self.cfg = config
        self.device = device
        self.image_size = self.cfg.dataset.image_size
        self.uv_size = self.cfg.model.uv_size
        self.flame_mask_file = self.cfg.model.flame_mask_path
        # self.cfg.model.tex_type = 'BFM'
        # self.cfg.model.n_tex = 50
        self._create_model(self.cfg.model)
        self._setup_renderer(self.cfg.model)

    def _setup_renderer(self, model_cfg):
        set_rasterizer(self.cfg.rasterizer_type)
        # self.render = SRenderY(self.image_size, obj_filename=model_cfg.topology_path, uv_size=model_cfg.uv_size,
                            #    rasterizer_type=self.cfg.rasterizer_type).to(self.device)
        self.render = SRenderY(self.image_size, flame_mask_file=self.flame_mask_file, obj_filename=model_cfg.topology_path, uv_size=model_cfg.uv_size,
        # self.render = SRenderY(self.image_size, obj_filename=model_cfg.topology_path, uv_size=model_cfg.uv_size,
                               rasterizer_type=self.cfg.rasterizer_type).to(self.device)
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

    def _create_model(self, model_cfg):
        # set up parameters
        self.n_param =  model_cfg.n_tex + model_cfg.n_exp + model_cfg.n_pose + model_cfg.n_cam + model_cfg.n_light
        self.n_detail = model_cfg.n_detail
        self.n_cond = model_cfg.n_exp + 3  # exp + jaw pose
        self.num_list = [ model_cfg.n_tex, model_cfg.n_exp, model_cfg.n_pose, model_cfg.n_cam,
                         model_cfg.n_light]
        
        self.param_dict = {i: model_cfg.get('n_' + i) for i in model_cfg.param_list}

        # encoders
        self.E_flame = ResnetEncoder(outsize=59).to(self.device)
        self.E_flame_224 = ResnetEncoder(outsize=self.n_param+100).to(self.device)
        self.E_detail = ResnetEncoder(outsize=self.n_detail).to(self.device)
        # decoders
        self.flame = FLAME(model_cfg).to(self.device)
        if model_cfg.use_tex:
            self.flametex = FLAMETex(model_cfg).to(self.device)
        self.D_detail = Generator(latent_dim=self.n_detail + self.n_cond, out_channels=1, out_scale=model_cfg.max_z,
                                  sample_mode='bilinear').to(self.device)
        # resume model
        model_path = self.cfg.pretrained_modelpath
        model_path_224 = self.cfg.pretrained_modelpath_224
        if os.path.exists(model_path):
            print(f'trained model found. load {model_path}')
            checkpoint = torch.load(model_path)
            self.checkpoint = checkpoint
            util.copy_state_dict(self.E_flame.state_dict(), checkpoint['E_flame'])
            # util.copy_state_dict(self.E_detail.state_dict(), checkpoint['E_detail'])
            # util.copy_state_dict(self.D_detail.state_dict(), checkpoint['D_detail'])
        # else:
        #     print(f'please check model path: {model_path}')
        #     # exit()
        if os.path.exists(model_path_224):
            print(f'trained model found. load {model_path_224}')
            checkpoint = torch.load(model_path_224)
            util.copy_state_dict(self.E_flame_224.state_dict(), checkpoint['E_flame'])
            util.copy_state_dict(self.E_detail.state_dict(), checkpoint['E_detail'])
            util.copy_state_dict(self.D_detail.state_dict(), checkpoint['D_detail'])
        else:
            print(f'please check 224 model path: {model_path_224}')
            # exit()
        # eval mode
        self.E_flame.eval()
        self.E_flame_224.eval()
        self.E_detail.eval()
        self.D_detail.eval()

    def decompose_code(self, code, num_dict):
        ''' Convert a flattened parameter vector to a dictionary of parameters
        code_dict.keys() = ['shape', 'tex', 'exp', 'pose', 'cam', 'light']
        '''
        code_dict = {}
        start = 0
        for key in num_dict:
            end = start + int(num_dict[key])
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
    def encode(self, images_224,  use_detail=True):
        if use_detail:
            # use_detail is for training detail model, need to set coarse model as eval mode
            with torch.no_grad():
                parameters = self.E_flame(images_224)
                parameters_224 = self.E_flame_224(images_224)
        else:
            # t1 = time.time()
            parameters = self.E_flame(images_224)
            with torch.no_grad():
                parameters_224 = self.E_flame_224(images_224)
            # print("Encode time..", (time.time() - t1)*1000)
        codedict = self.decompose_code(parameters_224, self.param_dict)
        codedict['images'] = images_224
        # # 'tex', 'exp', 'pose', 'cam', 'light'
        # codedict['shape'] =  parameters_224[:, :100]

        codedict['exp'] = parameters[:,:50]
        codedict['pose'][:,3:] = parameters[:,50:53]

        # codedict['pose'][:,:3] = codedict['pose'][:,:3] + parameters[:,53:56]
        # codedict['cam'] = codedict['cam'][:,:3] + parameters[:,56:59]

        if use_detail:
            detailcode = self.E_detail(images_224)
            codedict['detail'] = detailcode
        if self.cfg.model.jaw_type == 'euler':
            posecode = codedict['pose']
            euler_jaw_pose = posecode[:, 3:].clone()  # x for yaw (open mouth), y for pitch (left ang right), z for roll
            posecode[:, 3:] = batch_euler2axis(euler_jaw_pose)
            codedict['pose'] = posecode
            codedict['euler_jaw_pose'] = euler_jaw_pose
        return codedict

    # @torch.no_grad()
    def decode(self, codedict, rendering=True, iddict=None, vis_lmk=True, return_vis=True, use_detail=True,
               render_orig=False, original_image=None, tform=None):
        images = codedict['images']
        batch_size = images.shape[0]

        ## decode
        verts, landmarks2d, landmarks3d, mp_landmark = self.flame(shape_params=codedict['shape'], expression_params=codedict['exp'],
                                                     pose_params=codedict['pose'])
        # albedo = self.flametex(codedict['tex'])

        if self.cfg.model.use_tex:
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
            ops = self.render(verts, trans_verts, albedo, codedict['light'], h=h, w=w, background=background)
            ## output
            opdict['grid'] = ops['grid']
            opdict['rendered_images'] = ops['images']
            opdict['alpha_images'] = ops['alpha_images']
            opdict['normal_images'] = ops['normal_images']
            opdict['v_colors'] = ops['v_colors']

        if self.cfg.model.use_tex:
            opdict['albedo'] = albedo

        if use_detail:
            uv_z = self.D_detail(torch.cat([codedict['pose'][:, 3:], codedict['exp'], codedict['detail']], dim=1))
            if iddict is not None:
                uv_z = self.D_detail(torch.cat([iddict['pose'][:, 3:], iddict['exp'], codedict['detail']], dim=1))
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
            shape_images_full, shape_images, _, grid, alpha_images = self.render.render_shape(verts, trans_verts, h=h, w=w,
                                                                           images=background, return_grid=True)
                                                                           
            if use_detail:
                predicted_detail_images = F.grid_sample(uv_texture, ops['grid'].detach(), align_corners=False)

                detail_normal_images = F.grid_sample(uv_detail_normals, grid, align_corners=False) * alpha_images
                shape_detail_images_full, shape_detail_images = self.render.render_shape(verts, trans_verts,
                                                            detail_normal_images=detail_normal_images, h=h, w=w,
                                                            images=background)
            # detail_normal_images = F.grid_sample(uv_detail_normals, grid, align_corners=False) * alpha_images
            # shape_detail_images = self.render.render_shape(verts, trans_verts,
            #                                                detail_normal_images=detail_normal_images, h=h, w=w,
            #                                                images=background)

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
                    # 'shape_detail_images': shape_detail_images,
                    'render_images': images*(1-opdict['v_colors'])+predicted_detail_images*opdict['v_colors'],#opdict['rendered_images'],
                }
            else:
                visdict = {
                    'inputs': images,
                    # 'landmarks2d': util.tensor_vis_landmarks(images, landmarks2d),
                    # 'mp_landmark': util.tensor_vis_landmarks(images, mp_landmark),
                    # 'landmarks3d': util.tensor_vis_landmarks(images, landmarks3d),
                    'shape_images': shape_images,
                    'render_images': images*(1-opdict['v_colors'])+opdict['rendered_images']*opdict['v_colors']#opdict['rendered_images'],
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
    #                    normal_map=normal_map)
    #     # upsample mesh, save detailed mesh
    #     # texture = texture[:, :, [2, 1, 0]]
    #     # normals = opdict['normals'][i].cpu().numpy()
    #     # displacement_map = opdict['displacement_map'][i].cpu().numpy().squeeze()
    #     # dense_vertices, dense_colors, dense_faces = util.upsample_mesh(vertices, normals, faces, displacement_map,
    #     #                                                                texture, self.dense_template)
    #     # util.write_obj(filename.replace('.obj', '_detail.obj'),
    #     #                dense_vertices,
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
            'E_flame': self.E_flame.state_dict(),
            'E_detail': self.E_detail.state_dict(),
            'D_detail': self.D_detail.state_dict()
        }
