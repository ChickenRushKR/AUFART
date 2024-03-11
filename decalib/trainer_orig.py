
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

from .datasets import build_datasets
# from .datasets import build_datasets_NoAug
torch.backends.cudnn.benchmark = True

class Trainer(object):
    def __init__(self, model, config=None, device='cuda:0'):
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

        # deca model
        self.deca = model.to(self.device)
        self.configure_optimizers()
        self.load_checkpoint()

        # initialize loss
        if self.train_detail:     
            self.mrf_loss = lossfunc.IDMRFLoss()
            self.face_attr_mask = util.load_local_mask(image_size=self.cfg.model.uv_size, mode='bbx')
        else:
            self.id_loss = lossfunc.VGGFace2Loss(pretrained_model=self.cfg.model.fr_model_path)      
        
        logger.add(os.path.join(self.cfg.output_dir, self.cfg.train.log_dir, 'train.log'))
        if self.cfg.train.write_summary:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=os.path.join(self.cfg.output_dir, self.cfg.train.log_dir))
    
    def configure_optimizers(self):
        if self.train_detail:
            self.opt = torch.optim.Adam(
                                list(self.deca.E_detail.parameters()) + \
                                list(self.deca.D_detail.parameters())  ,
                                lr=self.cfg.train.lr,
                                amsgrad=False)
        else:
            self.opt = torch.optim.Adam(
                                    self.deca.E_flame.parameters(),
                                    lr=self.cfg.train.lr,
                                    amsgrad=False)
    def load_checkpoint(self):
        model_dict = self.deca.model_dict()

        self.expression_net = ExpressionLossNet().to(self.device)
        emotion_checkpoint = torch.load(self.cfg.emotion_checkpoint)['state_dict']
        emotion_checkpoint['linear.0.weight'] = emotion_checkpoint['linear.weight']
        emotion_checkpoint['linear.0.bias'] = emotion_checkpoint['linear.bias']
        self.expression_net.load_state_dict(emotion_checkpoint, strict=False)
        self.expression_net.eval()
        # self.au_net = 
        # resume training, including model weight, opt, steps
        # import ipdb; ipdb.set_trace()
        if self.cfg.train.resume and os.path.exists(os.path.join(self.cfg.output_dir, 'model.tar')):
            print('True')
            checkpoint = torch.load(os.path.join(self.cfg.output_dir, 'model.tar'))
            for key in model_dict.keys():
                if key in checkpoint.keys():
                    util.copy_state_dict(model_dict[key], checkpoint[key])
            util.copy_state_dict(self.opt.state_dict(), checkpoint['opt'])
            self.global_step = checkpoint['global_step']
            logger.info(f"resume training from {os.path.join(self.cfg.output_dir, 'model.tar')}")
            logger.info(f"training start from step {self.global_step}")
        # load model weights only
        elif os.path.exists(self.cfg.pretrained_modelpath):
            checkpoint = torch.load(self.cfg.pretrained_modelpath)
            key = 'E_flame'
            util.copy_state_dict(model_dict[key], checkpoint[key])
            self.global_step = 0
        else:
            logger.info('model path not found, start training from scratch')
            self.global_step = 0

    def training_step(self, batch, batch_nb, training_type='coarse'):
        self.deca.train()
        self.deca.E_flame_224.eval()

        if self.train_detail:
            self.deca.E_flame.eval()
        # [B, K, 3, size, size] ==> [BxK, 3, size, size]
        images_224 = batch['image_224'].to(self.device); images_224 = images_224.view(-1, images_224.shape[-3], images_224.shape[-2], images_224.shape[-1])
        images = batch['image'].to(self.device); images = images.view(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        lmk = batch['landmark'].to(self.device); lmk = lmk.view(-1, lmk.shape[-2], lmk.shape[-1])
        lmk_dense = batch['landmark_dense'].to(self.device); lmk_dense = lmk_dense.view(-1, lmk_dense.shape[-2], lmk_dense.shape[-1])
        masks = batch['mask'].to(self.device);  masks = masks.view(-1,images.shape[-3], images.shape[-2], images.shape[-1])
        #-- encoder
        codedict = self.deca.encode(images,images_224,  use_detail=self.train_detail)

        batch_size = images.shape[0]

        ###--------------- training coarse model
        if not self.train_detail:
            #-- decoder
            # rendering = True if self.cfg.loss.photo>0 else False
            rendering = True
            opdict = self.deca.decode(codedict, rendering = rendering, vis_lmk=False, return_vis=False, use_detail=False)
            opdict['images'] = images
            opdict['lmk'] = lmk
            opdict['lmk_dense'] = lmk_dense

            if self.cfg.loss.photo > 0.:
                #------ rendering
                # mask
                mask_face_eye = F.grid_sample(self.deca.uv_face_eye_mask.expand(batch_size,-1,-1,-1), opdict['grid'].detach(), align_corners=False) 
                # images
                predicted_images = opdict['rendered_images']*mask_face_eye*opdict['alpha_images']
                opdict['predicted_images'] = predicted_images

            #### ----------------------- Losses
            losses = {}
            
            ############################# base shape
            predicted_landmarks = opdict['landmarks2d']
            predicted_landmarks_dense = opdict['mp_landmark']

            if  self.cfg.loss.lmk > 0:
                losses['landmark'] = lossfunc.landmark_HRNet_loss(predicted_landmarks, lmk) * self.cfg.loss.lmk
                losses['relative_loss'] = lossfunc.relative_landmark_loss(predicted_landmarks, lmk) * self.cfg.loss.relative_d

            if self.cfg.loss.lmk_dense>0 :
                losses['landmark_dense'] = lossfunc.weighted_landmark_loss(predicted_landmarks_dense, lmk_dense) * self.cfg.loss.lmk_dense

            if self.cfg.loss.eyed > 0.:
                losses['eye_distance'] = lossfunc.eyed_loss(predicted_landmarks_dense, lmk_dense)*self.cfg.loss.eyed
            if self.cfg.loss.lipd > 0.:
                losses['lip_distance'] = lossfunc.lipd_loss(predicted_landmarks_dense, lmk_dense)*self.cfg.loss.lipd
            # if self.cfg.loss.relative_landmark > 0.:
            #     losses['relative_landmark'] = lossfunc.relative_landmark_loss(predicted_landmarks_dense, lmk_dense)*self.cfg.loss.relative_landmark

            if self.cfg.loss.photo > 0.:
                if self.cfg.loss.useSeg:
                    masks = masks[:,None,:,:]
                else:
                    masks = mask_face_eye*opdict['alpha_images']
 
                losses['photometric_texture'] = (masks*(predicted_images - images).abs()).mean()*self.cfg.loss.photo

            # if self.cfg.loss.id > 0.:
            #     shading_images = self.deca.render.add_SHlight(opdict['normal_images'], codedict['light'].detach())
            #     albedo_images = F.grid_sample(opdict['albedo'].detach(), opdict['grid'], align_corners=False)
            #     overlay = albedo_images*shading_images*mask_face_eye + images*(1-mask_face_eye)
            #     losses['identity'] = self.id_loss(overlay, images) * self.cfg.loss.id
            if self.cfg.loss.expression >0:
                losses['expression'] = F.mse_loss(self.expression_net(opdict['rendered_images']), self.expression_net(images))*self.cfg.loss.expression

            losses['shape_reg'] = (torch.sum(codedict['shape']**2)/2)*self.cfg.loss.reg_shape
            losses['expression_reg'] = (torch.sum(codedict['exp']**2)/2)*self.cfg.loss.reg_exp
            losses['tex_reg'] = (torch.sum(codedict['tex']**2)/2)*self.cfg.loss.reg_tex
            losses['light_reg'] = ((torch.mean(codedict['light'], dim=2)[:,:,None] - codedict['light'])**2).mean()*self.cfg.loss.reg_light
            if self.cfg.model.jaw_type == 'euler':
                # import ipdb; ipdb.set_trace()
                # reg on jaw pose
                losses['reg_jawpose_roll'] = (torch.sum(codedict['euler_jaw_pose'][:,-1]**2)/2)*100.
                losses['reg_jawpose_close'] = (torch.sum(F.relu(-codedict['euler_jaw_pose'][:,0])**2)/2)*10.
        

        #########################################################
        all_loss = 0.
        losses_key = losses.keys()
        for key in losses_key:
            all_loss = all_loss + losses[key]
        losses['all_loss'] = all_loss
        return losses, opdict

    # def evaluate(self):
    #     ''' NOW validation
    #     '''
    #     os.makedirs(os.path.join(self.cfg.output_dir, 'NOW_validation'), exist_ok=True)
    #     savefolder = os.path.join(self.cfg.output_dir, 'NOW_validation', f'step_{self.global_step:08}')
    #     os.makedirs(savefolder, exist_ok=True)
    #     self.deca.eval()
    #     # run now validation images
    #     from .datasets.now import NoWDataset
    #     dataset = NoWDataset(scale=(self.cfg.dataset.scale_min + self.cfg.dataset.scale_max) / 2)
    #     dataloader = DataLoader(dataset, batch_size=8, shuffle=False,
    #                             num_workers=8,
    #                             pin_memory=True,
    #                             drop_last=False)
    #     faces = self.deca.flame.faces_tensor.cpu().numpy()
    #     for i, batch in enumerate(tqdm(dataloader, desc='now evaluation ')):
    #         images = batch['image'].to(self.device)
    #         images_224 = batch['image_224'].to(self.device)
    #         imagename = batch['imagename']
    #         with torch.no_grad():
    #             codedict = self.deca.encode(images, images_224)
    #             _, visdict = self.deca.decode(codedict)
    #             codedict['exp'][:] = 0.
    #             codedict['pose'][:] = 0.
    #             opdict, _ = self.deca.decode(codedict)
    #         # -- save results for evaluation
    #         verts = opdict['verts'].cpu().numpy()
    #
    #         landmark_51 = opdict['landmarks3d_world'][:, 17:]
    #         landmark_7 = landmark_51[:, [19, 22, 25, 28, 16, 31, 37]]
    #         landmark_7 = landmark_7.cpu().numpy()
    #         for k in range(images.shape[0]):
    #             os.makedirs(os.path.join(savefolder, imagename[k]), exist_ok=True)
    #             # save mesh
    #             util.write_obj(os.path.join(savefolder, f'{imagename[k]}.obj'), vertices=verts[k], faces=faces)
    #             # save 7 landmarks for alignment
    #             np.save(os.path.join(savefolder, f'{imagename[k]}.npy'), landmark_7[k])
    #             for vis_name in visdict.keys():  # ['inputs', 'landmarks2d', 'shape_images']:
    #                 if vis_name not in visdict.keys():
    #                     continue
    #                 # import ipdb; ipdb.set_trace()
    #                 image = util.tensor2image(visdict[vis_name][k])
    #                 name = imagename[k].split('/')[-1]
    #                 # print(os.path.join(savefolder, imagename[k], name + '_' + vis_name +'.jpg'))
    #                 cv2.imwrite(os.path.join(savefolder, imagename[k], name + '_' + vis_name + '.jpg'), image)
    #         # visualize results to check
    #         util.visualize_grid(visdict, os.path.join(savefolder, f'{i}.jpg'))
    #
    #     ## then please run main.py in https://github.com/soubhiksanyal/now_evaluation, it will take around 30min to get the metric results
    #     self.deca.train()
    #     # self.deca.
    #     self.deca.E_flame_224.eval()
    # def validation_step(self):
    #     self.deca.eval()
    #     try:
    #         batch = next(self.val_iter)
    #     except:
    #         self.val_iter = iter(self.val_dataloader)
    #         batch = next(self.val_iter)
    #     images = batch['image'].to(self.device); images = images.view(-1, images.shape[-3], images.shape[-2], images.shape[-1])
    #     with torch.no_grad():
    #         codedict = self.deca.encode(images)
    #         opdict, visdict = self.deca.decode(codedict)
    #     savepath = os.path.join(self.cfg.output_dir, self.cfg.train.val_vis_dir, f'{self.global_step:08}.jpg')
    #     grid_image = util.visualize_grid(visdict, savepath, return_gird=True)
    #     self.writer.add_image('val_images', (grid_image/255.).astype(np.float32).transpose(2,0,1), self.global_step)
    #     self.deca.train()

    # def evaluate(self):
    #     ''' NOW validation
    #     '''
    #     os.makedirs(os.path.join(self.cfg.output_dir, 'NOW_validation'), exist_ok=True)
    #     savefolder = os.path.join(self.cfg.output_dir, 'NOW_validation', f'step_{self.global_step:08}')
    #     os.makedirs(savefolder, exist_ok=True)
    #     self.deca.eval()
    #     # run now validation images
    #     from .datasets.now import NoWDataset
    #     dataset = NoWDataset(scale=(self.cfg.dataset.scale_min + self.cfg.dataset.scale_max)/2)
    #     dataloader = DataLoader(dataset, batch_size=8, shuffle=False,
    #                         num_workers=8,
    #                         pin_memory=True,
    #                         drop_last=False)
    #     faces = self.deca.flame.faces_tensor.cpu().numpy()
    #     for i, batch in enumerate(tqdm(dataloader, desc='now evaluation ')):
    #         images = batch['image'].to(self.device)
    #         imagename = batch['imagename']
    #         with torch.no_grad():
    #             codedict = self.deca.encode(images)
    #             _, visdict = self.deca.decode(codedict)
    #             codedict['exp'][:] = 0.
    #             codedict['pose'][:] = 0.
    #             opdict, _ = self.deca.decode(codedict)
    #         #-- save results for evaluation
    #         verts = opdict['verts'].cpu().numpy()
    #         landmark_51 = opdict['landmarks3d_world'][:, 17:]
    #         landmark_7 = landmark_51[:,[19, 22, 25, 28, 16, 31, 37]]
    #         landmark_7 = landmark_7.cpu().numpy()
    #         for k in range(images.shape[0]):
    #             os.makedirs(os.path.join(savefolder, imagename[k]), exist_ok=True)
    #             # save mesh
    #             util.write_obj(os.path.join(savefolder, f'{imagename[k]}.obj'), vertices=verts[k], faces=faces)
    #             # save 7 landmarks for alignment
    #             np.save(os.path.join(savefolder, f'{imagename[k]}.npy'), landmark_7[k])
    #             for vis_name in visdict.keys(): #['inputs', 'landmarks2d', 'shape_images']:
    #                 if vis_name not in visdict.keys():
    #                     continue
    #                 # import ipdb; ipdb.set_trace()
    #                 image = util.tensor2image(visdict[vis_name][k])
    #                 name = imagename[k].split('/')[-1]
    #                 # print(os.path.join(savefolder, imagename[k], name + '_' + vis_name +'.jpg'))
    #                 cv2.imwrite(os.path.join(savefolder, imagename[k], name + '_' + vis_name +'.jpg'), image)
    #         # visualize results to check
    #         util.visualize_grid(visdict, os.path.join(savefolder, f'{i}.jpg'))
    #
    #     ## then please run main.py in https://github.com/soubhiksanyal/now_evaluation, it will take around 30min to get the metric results
    #     self.deca.train()

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
                    shape_images = self.deca.render.render_shape(opdict['verts'][visind], opdict['trans_verts'][visind])
                    visdict = {
                        # 'inputs': opdict['images'][visind],
                        # 'landmarks2d_gt': util.tensor_vis_landmarks(opdict['images'][visind], opdict['lmk'][visind], isScale=True),
                        'landmarks2d': util.tensor_vis_landmarks(opdict['images'][visind], opdict['landmarks2d'][visind], isScale=True),
                        # 'landmarks_dens_gt': util.tensor_vis_landmarks(opdict['images'][visind], opdict['lmk_dense'][visind], isScale=True),
                        # 'landmarks_dens': util.tensor_vis_landmarks(opdict['images'][visind], opdict['mp_landmark'][visind], isScale=True),
                        'shape_images': shape_images,
                        'rendered_images': opdict['rendered_images']
                        # 'predicted_images': opdict['predicted_images'][visind],
                    }
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
                    model_dict = self.deca.model_dict()
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
                self.opt.zero_grad(); all_loss.backward(); self.opt.step()
                self.global_step += 1
                if self.global_step > self.cfg.train.max_steps:
                    break