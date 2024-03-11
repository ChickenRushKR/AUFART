
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import cv2
from loguru import logger
from datetime import datetime
from tqdm import tqdm

from .utils import util
from .utils.config import cfg
from .utils import lossfunc2 as lossfunc
from .models.expression_loss import ExpressionLossNet
from .models.OpenGraphAU.model.MEFL import MEFARG
from .models.OpenGraphAU.utils import load_state_dict
from .models.OpenGraphAU.utils import *
from .models.OpenGraphAU.conf import get_config,set_logger,set_outdir,set_env
from .datasets import build_datasets_detail as build_datasets
# from .datasets import build_datasets_NoAug
torch.backends.cudnn.benchmark = True

class Trainer(object):
    def __init__(self, model, config=None, device='cuda:1'):
        if config is None:
            self.cfg = cfg
        else:
            self.cfg = config
        self.device = device
        self.batch_size = self.cfg.dataset.batch_size
        self.image_size = self.cfg.dataset.image_size
        self.uv_size = self.cfg.model.uv_size
        self.K = self.cfg.dataset.K
        # training stage: coarse and detail
        self.train_detail = self.cfg.train.train_detail
        self.vis_au = self.cfg.train.vis_au

        # mymodel model
        self.mymodel = model.to(self.device)
        self.configure_optimizers()
        self.load_checkpoint()

        self.au_weight = au_weights()
        # if self.cfg.loss.weightedAU:
            # self.weightedAU = np.load('/mnt/hdd/EncoderTrainingCode/Code/data/AU_weight.npy')
            # self.weightedAU = torch.tensor(self.weightedAU).to(self.device)
            # self.au_weight = au_weights()

        # initialize loss
        # if self.train_detail:     
        # self.mrf_loss = lossfunc.IDMRFLoss().eval(); self.mrf_loss.requires_grad_(False)
        self.ffloss = lossfunc.FocalFrequencyLoss()
        self.face_attr_mask = util.load_local_mask(image_size=self.cfg.model.uv_size, mode='bbx')
        # else:
        #     self.id_loss = lossfunc.VGGFace2Loss(pretrained_model=self.cfg.model.fr_model_path)      
        
        logger.add(os.path.join(self.cfg.output_dir, self.cfg.train.log_dir, 'train.log'))
        if self.cfg.train.write_summary:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=os.path.join(self.cfg.output_dir, self.cfg.train.log_dir))
    
    def configure_optimizers(self):
        self.opt = torch.optim.Adam(
                                # list(self.mymodel.GATE.parameters()) + list(self.mymodel.AU_Encoder.parameters()),\
                                # list(self.mymodel.GATE_detail.parameters()) +
                                # list(self.mymodel.AUD_Encoder.parameters()) +
                                list(self.mymodel.E_detail.parameters()) +
                                list(self.mymodel.D_detail.parameters())  ,
                                lr=self.cfg.train.lr,
                                amsgrad=False)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, step_size=self.cfg.train.stepLR_steps, gamma=0.999)
    def load_checkpoint(self):
        # au config
        self.auconf = get_config()
        self.auconf.evaluate = True
        self.auconf.gpu_ids = os.environ["CUDA_VISIBLE_DEVICES"]
        set_env(self.auconf)
        model_dict = self.mymodel.model_dict()

        self.expression_net = ExpressionLossNet().to(self.device)
        emotion_checkpoint = torch.load(self.cfg.emotion_checkpoint)['state_dict']
        emotion_checkpoint['linear.0.weight'] = emotion_checkpoint['linear.weight']
        emotion_checkpoint['linear.0.bias'] = emotion_checkpoint['linear.bias']
        self.expression_net.load_state_dict(emotion_checkpoint, strict=False)
        self.expression_net.eval()
        self.AU_net = MEFARG(num_main_classes=self.auconf.num_main_classes, num_sub_classes=self.auconf.num_sub_classes, backbone=self.auconf.arc).to(self.device)
        self.AU_net = load_state_dict(self.AU_net, self.auconf.resume).to(self.device)
        self.AU_net.eval()
        # resume training, including model weight, opt, steps
        # import ipdb; ipdb.set_trace()
        # load model weights only
        if os.path.exists(self.cfg.pretrained_modelpath):
            checkpoint = torch.load(self.cfg.pretrained_modelpath)
            for key in model_dict.keys():
                if key in checkpoint.keys():
                    util.copy_state_dict(model_dict[key], checkpoint[key])
        if self.cfg.train.resume and os.path.exists(os.path.join(self.cfg.output_dir, 'model.tar')):
            # print('True')
            checkpoint = torch.load(os.path.join(self.cfg.output_dir, 'model.tar')) 
            model_dict = self.mymodel.model_dict()
            for key in model_dict.keys():
                if key in checkpoint.keys():
                    print(key)
                    util.copy_state_dict(model_dict[key], checkpoint[key])
                else:
                    print("check model path", os.path.join(self.cfg.output_dir, 'model.tar'))
                    exit()
            util.copy_state_dict(self.opt.state_dict(), checkpoint['opt'])
            # self.opt.param_groups[0]['lr'] = 0.000005
            self.global_step = checkpoint['global_step']
            logger.info(f"resume training from {os.path.join(self.cfg.output_dir, 'model.tar')}")
            logger.info(f"training start from step {self.global_step}")
        else:
            logger.info('model path not found, start training from scratch')
            self.global_step = 0
        if os.path.exists(self.cfg.pretrained_modelpath_224) and not os.path.exists(os.path.join(self.cfg.output_dir, 'model.tar')):
                # model_dict = self.mymodel.model_dict()
            logger.info(f'detail load fine tuning')
            checkpoint = torch.load(self.cfg.pretrained_modelpath_224)
            # util.copy_state_dict(model_dict['E_detail'], checkpoint['E_detail'])
            # util.copy_state_dict(model_dict['D_detail'], checkpoint['D_detail'])

    def training_step(self, batch, batch_nb, training_type='coarse'):
        self.mymodel.train()
        self.mymodel.GATE.eval()
        self.mymodel.AU_Encoder.eval()
        self.mymodel.AUNet.eval()
        self.mymodel.E_flame.eval()
        self.mymodel.E_detail.train()
        # self.mymodel.GATE_detail.train()
        # self.mymodel.AUD_Encoder.train()
        self.mymodel.D_detail.train()

        # [B, K, 3, size, size] ==> [BxK, 3, size, size]
        images_224 = batch['image_224'].to(self.device); images_224 = images_224.view(-1, images_224.shape[-3], images_224.shape[-2], images_224.shape[-1])
        # images = batch['image'].to(self.device); images = images.view(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        lmk = batch['landmark'].to(self.device); lmk = lmk.view(-1, lmk.shape[-2], lmk.shape[-1])
        lmk_dense = batch['landmark_dense'].to(self.device); lmk_dense = lmk_dense.view(-1, lmk_dense.shape[-2], lmk_dense.shape[-1])
        masks = batch['mask'].to(self.device); masks = masks.view(-1, images_224.shape[-2], images_224.shape[-1])
        # masks = batch['mask'].to(self.device); masks = masks.view(-1,images_224.shape[-3], images_224.shape[-2], images_224.shape[-1])
        # masks = batch['mask'].to(self.device); masks = masks.view(-1,images.shape[-3], images.shape[-2], images.shape[-1])
        #-- encoder
        codedict = self.mymodel.encode(images_224,  use_detail=self.train_detail)
        images = images_224
        batch_size = images_224.shape[0]

        #-- decoder
        shapecode = codedict['shape']
        expcode = codedict['exp']
        posecode = codedict['pose']
        texcode = codedict['tex']
        lightcode = codedict['light']
        detailcode = codedict['detail']
        aucode = codedict['au']
        cam = codedict['cam']

        # FLAME - world space
        verts, landmarks2d, landmarks3d, mp_landmark = self.mymodel.flame(shape_params=shapecode, expression_params=expcode, pose_params=posecode)
        landmarks2d = util.batch_orth_proj(landmarks2d, codedict['cam'])[:,:,:2]; landmarks2d[:,:,1:] = -landmarks2d[:,:,1:] #; landmarks2d = landmarks2d*self.image_size/2 + self.image_size/2
        mp_landmark = util.batch_orth_proj(mp_landmark, codedict['cam'])[:,:,:2]; mp_landmark[:,:,1:] = -mp_landmark[:,:,1:]
        # world to camera
        trans_verts = util.batch_orth_proj(verts, cam)
        predicted_landmarks = util.batch_orth_proj(landmarks2d, cam)[:,:,:2]
        # camera to image space
        trans_verts[:,:,1:] = -trans_verts[:,:,1:]
        predicted_landmarks[:,:,1:] = - predicted_landmarks[:,:,1:]
        
        albedo = self.mymodel.flametex(texcode)

        #------ rendering
        ops = self.mymodel.render(verts, trans_verts, albedo, lightcode) 
        # mask
        mask_face_eye = F.grid_sample(self.mymodel.uv_face_eye_mask.expand(batch_size,-1,-1,-1), ops['grid'].detach(), align_corners=False)
        # images
        predicted_images = ops['images']*mask_face_eye*ops['alpha_images']

        masks = masks[:,None,:,:]

        # uv_z = self.mymodel.D_detail(torch.cat([posecode[:,3:], expcode, detailcode, aucode[:,:27]], dim=1))
        uv_z = self.mymodel.D_detail(torch.cat([posecode[:,3:], expcode, detailcode], dim=1))
        # render detail
        uv_detail_normals = self.mymodel.displacement2normal(uv_z, verts, ops['normals'])
        uv_shading = self.mymodel.render.add_SHlight(uv_detail_normals, lightcode.detach())
        uv_texture = albedo.detach()*uv_shading
        predicted_detail_images = F.grid_sample(uv_texture, ops['grid'].detach(), align_corners=False)

        #--- extract texture
        uv_pverts = self.mymodel.render.world2uv(trans_verts).detach()
        uv_gt = F.grid_sample(torch.cat([images, masks], dim=1), uv_pverts.permute(0,2,3,1)[:,:,:,:2], mode='bilinear', align_corners=False)
        uv_texture_gt = uv_gt[:,:3,:,:].detach(); uv_mask_gt = uv_gt[:,3:,:,:].detach()
        # self-occlusion
        normals = util.vertex_normals(trans_verts, self.mymodel.render.faces.expand(batch_size, -1, -1))
        uv_pnorm = self.mymodel.render.world2uv(normals)
        uv_mask = (uv_pnorm[:,[-1],:,:] < -0.05).float().detach()
        ## combine masks
        uv_vis_mask = uv_mask_gt*uv_mask*self.mymodel.uv_face_mask
        
        #### ----------------------- Losses
        losses = {}
        ############################### details
        # if self.cfg.loss.old_mrf: 
        #     if self.cfg.loss.old_mrf_face_mask:
        #         masks = masks*mask_face_eye*ops['alpha_images']
        #     losses['photo_detail'] = (masks*(predicted_detailed_image - images).abs()).mean()*100
        #     losses['photo_detail_mrf'] = self.mrf_loss(masks*predicted_detailed_image, masks*images)*0.1
        # else:
        pi = 0
        new_size = 256
        uv_texture_patch = F.interpolate(uv_texture[:, :, self.face_attr_mask[pi][2]:self.face_attr_mask[pi][3], self.face_attr_mask[pi][0]:self.face_attr_mask[pi][1]], [new_size, new_size], mode='bilinear')
        uv_texture_gt_patch = F.interpolate(uv_texture_gt[:, :, self.face_attr_mask[pi][2]:self.face_attr_mask[pi][3], self.face_attr_mask[pi][0]:self.face_attr_mask[pi][1]], [new_size, new_size], mode='bilinear')
        uv_vis_mask_patch = F.interpolate(uv_vis_mask[:, :, self.face_attr_mask[pi][2]:self.face_attr_mask[pi][3], self.face_attr_mask[pi][0]:self.face_attr_mask[pi][1]], [new_size, new_size], mode='bilinear')
        
        detail_normal_images = F.grid_sample(uv_detail_normals, ops['grid'], align_corners=False) * ops['alpha_images']
        shape_detail_images_full, shape_detail_images = self.mymodel.render.render_shape(verts, trans_verts,
                                                    detail_normal_images=detail_normal_images,# h=h, w=w,
                                                    images=images)
        losses['photo_detail'] = (uv_texture_patch*uv_vis_mask_patch - uv_texture_gt_patch*uv_vis_mask_patch).abs().mean()*self.cfg.loss.photo_D
        # losses['photo_detail_mrf'] = self.mrf_loss(uv_texture_patch*uv_vis_mask_patch, uv_texture_gt_patch*uv_vis_mask_patch)*self.cfg.loss.photo_D*self.cfg.loss.mrf
        losses['FFLoss'] = lossfunc.FFLoss(uv_texture_patch*uv_vis_mask_patch, uv_texture_gt_patch*uv_vis_mask_patch)*self.cfg.loss.photo_D*self.cfg.loss.mrf

        losses['z_reg'] = torch.sum(uv_z.abs())*self.cfg.loss.reg_z
        losses['z_diff'] = lossfunc.shading_smooth_loss(uv_shading)*self.cfg.loss.reg_diff
        if self.cfg.loss.reg_sym > 0.:
            nonvis_mask = (1 - util.binary_erosion(uv_vis_mask))
            losses['z_sym'] = (nonvis_mask*(uv_z - torch.flip(uv_z, [-1]).detach()).abs()).sum()*self.cfg.loss.reg_sym
        if self.cfg.loss.au_D > 0.:
            rend_au_loss = self.AU_net(predicted_detail_images)
            image_au_loss = self.AU_net(images)
            # image_au_loss = aucode
            # rend_au_loss = self.mymodel.AUNet(predicted_detail_images, use_gnn=True)[2]
            # losses['au_D'] = F.mse_loss(rend_au_loss[2].flatten(), image_au_loss[2].flatten())#*self.cfg.loss.au_D
            losses['au_D'] = F.mse_loss(rend_au_loss[2].flatten(), image_au_loss[2].flatten())*self.cfg.loss.au_D
            # losses['au_D'] = (1 - F.cosine_similarity(rend_au_loss[2].flatten(), image_au_loss[2].flatten(), dim=0))*self.cfg.loss.au_D

        opdict = {
            'verts': verts,
            'trans_verts': trans_verts,
            'landmarks2d': landmarks2d,
            'mp_landmark': mp_landmark,
            'predicted_images': predicted_images,
            'predicted_detail_images': predicted_detail_images,
            'shape_detail_images': shape_detail_images,
            'images': images,
            'lmk': lmk,
            'lmk_dense':lmk_dense,
        }
        if self.vis_au:
            opdict['au_img'] = image_au_loss[1]
            opdict['au_rend'] = rend_au_loss[1]

        #########################################################
        all_loss = 0.
        losses_key = losses.keys()
        for key in losses_key:
            all_loss = all_loss + losses[key]
        losses['all_loss'] = all_loss
        return losses, opdict
    
    def prepare_data(self):
        # self.train_dataset = build_datasets_NoAug.build_train(self.cfg.dataset)
        self.train_dataset = build_datasets.build_train(self.cfg.dataset)
        logger.info('---- training data numbers: ', len(self.train_dataset))

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                            num_workers=self.cfg.dataset.num_workers,
                            pin_memory=True,
                            drop_last=True)
        self.train_iter = iter(self.train_dataloader)
        # self.val_dataloader = DataLoader(self.val_dataset, batch_size=8, shuffle=True,
        #                     num_workers=8,
        #                     pin_memory=True,
        #                     drop_last=False)
        # self.val_iter = iter(self.val_dataloader)

    def fit(self):
        self.prepare_data()

        iters_every_epoch = int(len(self.train_dataset)/self.batch_size)
        start_epoch = self.global_step//iters_every_epoch
        for epoch in range(start_epoch, self.cfg.train.max_epochs):
            # random.shuffle(self.train_dataset)
            # for step, batch in enumerate(tqdm(self.train_dataloader, desc=f"Epoch: {epoch}/{self.cfg.train.max_epochs}")):
            for step in tqdm(range(iters_every_epoch), desc=f"Epoch[{epoch+1}/{self.cfg.train.max_epochs}]"):
                if epoch*iters_every_epoch + step < self.global_step:
                    continue
                try:
                    batch = next(self.train_iter)
                except:
                    self.train_iter = iter(self.train_dataloader)
                    batch = next(self.train_iter)
                losses, opdict = self.training_step(batch, step)
                if self.global_step % self.cfg.train.log_steps == 0:
                    loss_info = f"ExpName: {self.cfg.exp_name} \nEpoch: {epoch}, Iter: {step}/{iters_every_epoch}, Time: {datetime.now().strftime('%Y-%m-%d-%H:%M:%S')} \n"
                    for k, v in losses.items():
                        loss_info = loss_info + f'{k}: {v:.4f}, '
                        if self.cfg.train.write_summary:
                            self.writer.add_scalar('train_loss/'+k, v, global_step=self.global_step)                    
                    logger.info(loss_info)
                    if self.global_step % self.cfg.train.vis_steps == 0 or step == (iters_every_epoch-1):
                        visind = list(range(self.cfg.dataset.batch_size* self.cfg.dataset.K)) #!!!!!!
                        # visind = list(range(self.cfg.dataset.batch_size))
                        shape_images_full, shape_images_face = self.mymodel.render.render_shape(opdict['verts'][visind], opdict['trans_verts'][visind], images=opdict['images'][visind])
                        if self.vis_au:
                            visdict = {
                                'inputs': opdict['images'][visind],
                                # 'landmarks2d_gt': util.tensor_vis_landmarks(opdict['images'][visind], opdict['lmk'][visind], isScale=True),
                                # 'landmarks2d': util.tensor_vis_landmarks(opdict['images'][visind], opdict['landmarks2d'][visind], isScale=True),
                                # 'landmarks_dens': util.tensor_vis_landmarks(opdict['images'][visind], opdict['mp_landmark'][visind], opdict['lmk_dense'][visind], isScale=True),
                                
                                # 'landmarks_dens': util.tensor_vis_landmarks(opdict['images'][visind], opdict['mp_landmark'][visind], isScale=True),
                                # 'inputs': opdict['images'][visind],
                                # 'au_gt': util.draw_activation_circles(opdict['images'][visind], opdict['lmk_dense'][visind], opdict['au_img'], self.au_weight),
                                'predicted_detail_images': opdict['predicted_detail_images'][visind],
                                'shape_detail_images': opdict['shape_detail_images'][visind],
                                'shape_images': shape_images_face,
                                'vis_au': util.vis_au(opdict['au_img'], opdict['au_rend']),
                                # 'shape_images_full': shape_images_full,
                                # 'au_rend': util.draw_activation_circles(opdict['predicted_detail_images'][visind], opdict['mp_landmark'][visind], opdict['au_rend'], self.au_weight),
                                # 'rendered_images': opdict['rendered_images']
                                # 'predicted_images': opdict['predicted_images'][visind],
                            }
                            if 'predicted_detail_images' in opdict.keys():
                                visdict['predicted_detail_images'] = opdict['predicted_detail_images'][visind]
                        else:
                            visdict = {
                                'inputs': opdict['images'][visind],
                                # 'landmarks2d_gt': util.tensor_vis_landmarks(opdict['images'][visind], opdict['lmk'][visind], isScale=True),
                                # 'landmarks2d': util.tensor_vis_landmarks(opdict['images'][visind], opdict['landmarks2d'][visind], isScale=True),
                                # 'landmarks_dens_gt': util.tensor_vis_landmarks(opdict['images'][visind], opdict['lmk_dense'][visind], isScale=True),
                                # 'landmarks_dens': util.tensor_vis_landmarks(opdict['images'][visind], opdict['mp_landmark'][visind], isScale=True),
                                'predicted_detail_images': opdict['predicted_detail_images'][visind],
                                'shape_detail_images': opdict['shape_detail_images'][visind],
                                'shape_images': shape_images_face,
                                # 'shape_images_full': shape_images_full,
                                # 'rendered_images': opdict['rendered_images']
                                # 'predicted_images': opdict['predicted_images'][visind],
                            }
                            if 'predicted_detail_images' in opdict.keys():
                                visdict['predicted_detail_images'] = opdict['predicted_detail_images'][visind]
                        # if 'predicted_images' in opdict.keys():
                        #     visdict['predicted_images'] = opdict['predicted_images'][visind]
                        # if 'predicted_detail_images' in opdict.keys():
                        #     visdict['predicted_detail_images'] = opdict['predicted_detail_images'][visind]

                        savepath = os.path.join(self.cfg.output_dir, self.cfg.train.vis_dir, f'{self.global_step:06}.jpg')
                        grid_image = util.visualize_grid(visdict, savepath, return_gird=True)
                        # import ipdb; ipdb.set_trace()                    
                        # self.writer.add_image('train_images', (grid_image/255.).astype(np.float32).transpose(2,0,1), self.global_step)

                        print("epoch and step:", epoch, step, self.global_step)
                    
                if self.global_step>0 and self.global_step % self.cfg.train.checkpoint_steps == 0 or step == (iters_every_epoch-1):
                    model_dict = self.mymodel.model_dict()
                    # model_dict = {key: model_dict[key]}
                    model_dict['opt'] = self.opt.state_dict()
                    model_dict['global_step'] = self.global_step
                    model_dict['batch_size'] = self.batch_size
                    torch.save(model_dict, os.path.join(self.cfg.output_dir, 'model' + '.tar'))
                    # 
                    if self.global_step % self.cfg.train.checkpoint_steps*10 == 0 or step == (iters_every_epoch-1):
                        os.makedirs(os.path.join(self.cfg.output_dir, 'models'), exist_ok=True)
                        torch.save(model_dict, os.path.join(self.cfg.output_dir, 'models', f'{self.global_step:08}.tar'))   
                #
                # if self.global_step % self.cfg.train.val_steps == 0:
                #     self.validation_step()
                #
                # if self.global_step % self.cfg.train.eval_steps == 0:
                #     self.evaluate()

                all_loss = losses['all_loss']
                self.opt.zero_grad(); all_loss.backward(); self.opt.step(); self.scheduler.step();
                self.global_step += 1
                if self.global_step > self.cfg.train.max_steps:
                    break

class au_weights:
    def __init__(self): 
        # landmark index
        # brow: 0~19
        self.brow = [i for i in range(0, 20)]
        # eye_low: 21, 22, 28, 29, 30, 31, 32, 34, 35, 37, 38, 44, 45, 46, 47, 48, 50, 51
        self.eye_low = [21, 22, 28, 29, 30, 31, 32, 34, 35, 37, 38, 44, 45, 46, 47, 48, 50, 51]
        # eye_up: 20, 21, 22, 23, 24, 25, 26, 27, 33, 36, 37, 38, 39, 40, 41, 42, 43, 49
        self.eye_up = [20, 21, 22, 23, 24, 25, 26, 27, 33, 36, 37, 38, 39, 40, 41, 42, 43, 49]
        # eye_all: 20~51
        self.eye_all = [i for i in range(20, 52)]
        # nose: 52~64
        self.nose = [i for i in range(52, 65)]
        # lip_up: 65, 66, 69, 70, 71, 72, 73, 74, 75, 76, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 103, 104
        self.lip_up = [65, 66, 69, 70, 71, 72, 73, 74, 75, 76, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 103, 104]
        # lip_end: 71, 72, 73, 74, 79, 80, 81, 82, 85, 86, 88, 89, 90, 91, 92, 93, 97, 98, 99, 100, 103, 104
        self.lip_end = [71, 72, 73, 74, 79, 80, 81, 82, 85, 86, 88, 89, 90, 91, 92, 93, 97, 98, 99, 100, 103, 104]
        # mouth: 65~104
        self.mouth = [i for i in range(65, 105)]

        # AU index
        self.brow_au = [0, 1, 2, 27, 28, 29, 30, 31, 32]
        self.eye_low_au = [3]
        self.eye_up_au = [4, 33, 34]
        self.eye_all_au = [5]
        self.nose_au = [6, 8, 25, 26, 37, 38]
        self.lip_up_au = [7, 8, 35, 36, 37, 38]
        self.lip_end_au = [9, 10, 11, 12, 13, 39, 40]
        self.mouth_au = [i for i in range(14, 25)]

    def lmk_to_au(self, idx):
        related_au = []
        if idx in self.brow:
            related_au += self.brow_au
        if idx in self.eye_low:
            related_au += self.eye_low_au
        if idx in self.eye_up:
            related_au += self.eye_up_au
        if idx in self.eye_all:
            related_au += self.eye_all_au
        if idx in self.nose:
            related_au += self.nose_au
        if idx in self.lip_up:
            related_au += self.lip_up_au
        if idx in self.lip_end:
            related_au += self.lip_end_au
        if idx in self.mouth:
            related_au += self.mouth_au
        return list(set(related_au))