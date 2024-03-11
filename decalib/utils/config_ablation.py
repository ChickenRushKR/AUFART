'''
Default config for DECA
'''
from yacs.config import CfgNode as CN
import argparse
import yaml
import os

cfg = CN()

abs_deca_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
cfg.deca_dir = abs_deca_dir
cfg.device = 'cuda:1'
cfg.device_id = '0,1'

cfg.pretrained_modelpath = "/mnt/hdd/EncoderTrainingCode/Code/data/model_new.tar"
# cfg.pretrained_modelpath = "/mnt/hdd/EncoderTrainingCode/Code/data/model_new.tar"
cfg.pretrained_modelpath_224 = "/mnt/hdd/EncoderTrainingCode/Code/data/deca_model.tar"
cfg.emotion_checkpoint ='/mnt/hdd/EncoderTrainingCode/Code/data/dataloader_idx_0=1.27607644.ckpt'

cfg.output_dir = ''
cfg.rasterizer_type = 'pytorch3d' #
# ---------------------------------------------------------------------------- #
# Options for Face model
# ---------------------------------------------------------------------------- #
cfg.model = CN()
cfg.model.topology_path = '/mnt/hdd/EncoderTrainingCode/Code/data/head_template.obj'
# texture data original from http://files.is.tue.mpg.de/tbolkart/FLAME/FLAME_texture_data.zip
cfg.model.dense_template_path = '/mnt/hdd/EncoderTrainingCode/Code/data/texture_data_256.npy'
cfg.model.fixed_displacement_path = '/mnt/hdd/EncoderTrainingCode/Code/data/fixed_displacement_256.npy'
cfg.model.flame_model_path = '/mnt/hdd/EncoderTrainingCode/Code/data/generic_model.pkl'
# cfg.model.flame_model_path = '/mnt/hdd/EncoderTrainingCode/Code/data/flame2023.pkl' # FLAME 2023
cfg.model.flame_lmk_embedding_path = '/mnt/hdd/EncoderTrainingCode/Code/data/landmark_embedding.npy'
cfg.model.mp_lmk_embedding_path = '/mnt/hdd/EncoderTrainingCode/Code/data/mediapipe_landmark_embedding.npz'
cfg.model.flame_mask_path = '/home/cine/LGAI/Final_Code/TestPart/data/FLAME_masks.pkl'
cfg.model.face_mask_path = '/mnt/hdd/EncoderTrainingCode/Code/data/uv_face_mask.png'
cfg.model.face_eye_mask_path = '/mnt/hdd/EncoderTrainingCode/Code/data/uv_face_eye_mask.png'
cfg.model.mean_tex_path = '/mnt/hdd/EncoderTrainingCode/Code/data/mean_texture.jpg'
cfg.model.flame_tex_path = '/mnt/hdd/EncoderTrainingCode/Code/data/FLAME_texture.npz'
cfg.model.tex_path = '/mnt/hdd/EncoderTrainingCode/Code/data/FLAME_albedo_from_BFM.npz'
cfg.model.pretrained_modelpath_albedo = "/mnt/hdd/EncoderTrainingCode/Code/data/TRUST_models_BalanceAlb_version/E_albedo_BalanceAlb.tar"
cfg.model.pretrained_modelpath_facel = "/mnt/hdd/EncoderTrainingCode/Code/data/TRUST_models_BalanceAlb_version/E_face_light_BalanceAlb.tar"
cfg.model.pretrained_modelpath_scene = "/mnt/hdd/EncoderTrainingCode/Code/data/TRUST_models_BalanceAlb_version/E_scene_light_BalanceAlb.tar"
cfg.model.lightprobe_normal_path = "/mnt/hdd/EncoderTrainingCode/Code/data/lightprobe_normal_images.npy"
cfg.model.lightprobe_albedo_path = "/mnt/hdd/EncoderTrainingCode/Code/data/lightprobe_albedo_images.npy"
cfg.model.BalanceAlb_tex_path= './data/BalanceAlb_model.npz'
cfg.model.BFM_tex_path = '/mnt/hdd/EncoderTrainingCode/Code/data/FLAME_texture.npz'
cfg.model.tex_type = 'BFM' # BFM, FLAME, albedoMM, BalanceAlb
cfg.model.uv_size = 256
cfg.model.image_size = 224
cfg.model.param_list = ['shape', 'tex', 'exp', 'pose', 'cam', 'light']
cfg.model.n_shape = 100
# cfg.model.param_list = [ 'tex', 'exp', 'pose', 'cam', 'light']
cfg.model.n_tex = 50
cfg.model.n_exp = 50
cfg.model.n_cam = 3
cfg.model.n_pose = 6
cfg.model.n_light = 27
cfg.model.use_tex = True
# cfg.model.n_facelight = 27
# cfg.model.n_scenelight = 3

# cfg.model.jaw_type = 'aa' # default use axis angle, another option: euler. Note that: aa is not stable in the beginning
cfg.model.jaw_type = 'aa' # default use axis angle, another option: euler. Note that: aa is not stable in the beginning
# face recognition model
cfg.model.fr_model_path = '/mnt/hdd/EncoderTrainingCode/Code/data/resnet50_ft_weight.pkl'
cfg.model.extract_tex = True

## details
cfg.model.n_detail = 128
cfg.model.max_z = 0.01

# ---------------------------------------------------------------------------- #
# Options for Dataset
# ---------------------------------------------------------------------------- #
cfg.dataset = CN()
cfg.dataset.training_data = ['selfDataset']

cfg.dataset.mediapipePath = '/mnt/hdd/EncoderTrainingCode/Code/data/mediapipe_landmark_embedding.npz'
cfg.dataset.batch_size = 1
cfg.dataset.K = 1
cfg.dataset.isSingle = False
# cfg.dataset.isSingle = True
cfg.dataset.num_workers = 3
# cfg.dataset.image_size = 224
cfg.dataset.image_size = 224 # from 224 to 448
cfg.dataset.scale_min = 1.4
cfg.dataset.scale_max = 1.8
cfg.dataset.trans_scale = 0.
cfg.dataset.lmksize = 68

# ---------------------------------------------------------------------------- #
# Options for training
# ---------------------------------------------------------------------------- #
cfg.train = CN()
cfg.train.train_detail = False
cfg.train.max_epochs = 500
cfg.train.max_steps = 1000000
cfg.train.lr = 1e-4
cfg.train.log_dir = 'logs'
cfg.train.log_steps = 10
cfg.train.vis_dir = 'train_images'
cfg.train.vis_steps = 200
cfg.train.vis_au = False
cfg.train.stepLR_steps = 10000
# cfg.train.write_summary = True
cfg.train.write_summary = True
cfg.train.checkpoint_steps = 500
# cfg.train.val_steps = 500
# cfg.train.val_vis_dir = 'val_images'
cfg.train.eval_steps = 5000
cfg.train.resume = False

# ---------------------------------------------------------------------------- #
# Options for Losses
# ---------------------------------------------------------------------------- #
cfg.loss = CN()

cfg.loss.lmk = 1.

cfg.loss.useWlmk = True
cfg.loss.lmk_dense = 1.
cfg.loss.eyed = 0.06 # change to new loss and give new coefficient value
cfg.loss.relative_d = 0.5 # change to new loss and give new coefficient value
cfg.loss.lipd = 1.
# cfg.loss.photo = 2.0
cfg.loss.photo = 0.2 # version 6 is 2.0
cfg.loss.useSeg = True
cfg.loss.expression = 0.
cfg.loss.mainAU = 0.5
cfg.loss.subAU = 0.5
cfg.loss.weightedAU = False
cfg.loss.focalAU = True
cfg.loss.reg_exp = 1e-04
cfg.loss.reg_shape = 0
cfg.loss.reg_tex = 0#1e-04

cfg.loss.reg_pose = 1e-05
cfg.loss.reg_cam = 1e-05

cfg.loss.reg_light = 0#1.
cfg.loss.reg_jaw_pose = 0.8 #1.
cfg.loss.use_gender_prior = False

# # loss for detail
# cfg.loss.detail_consistency = True
# cfg.loss.useConstraint = True
cfg.loss.mrf = 5e-2
cfg.loss.photo_D = 2.
cfg.loss.reg_sym = 0.005
cfg.loss.reg_z = 0.005
cfg.loss.reg_diff = 0.005
cfg.loss.au_D = 0.0


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return cfg.clone()

def update_cfg(cfg, cfg_file):
    cfg.merge_from_file(cfg_file)
    return cfg.clone()

def parse_args(cfg_name=None):
    parser = argparse.ArgumentParser()
    # parser.add_argument('--cfg', type=str, help='cfg file path')
    # parser.add_argument('--cfg', type=str,default='configs/release_version/deca_pretrain.yml', help='cfg file path')
    if cfg_name == None:
        parser.add_argument('--cfg', type=str, default='configs/release_version/deca_coarse.yml', help='cfg file path')
    else:
        parser.add_argument('--cfg', type=str, default=cfg_name, help='cfg file path')
    parser.add_argument('--mode', type=str, default = 'train', help='deca mode')

    args = parser.parse_args()
    print(args, end='\n\n')

    cfg = get_cfg_defaults()
    cfg.mode = args.mode
    # import ipdb; ipdb.set_trace()
    if args.cfg is not None:
        cfg_file = args.cfg
        cfg = update_cfg(cfg, args.cfg)
        cfg.cfg_file = cfg_file

    return cfg
# def parse_args():
#     parser = argparse.ArgumentParser()
#     # parser.add_argument('--cfg', type=str,default='configs/release_version/deca_pretrain.yml', help='cfg file path')
#     parser.add_argument('--cfg', type=str,  help='cfg file path')
#     # parser.add_argument('--cfg', type=str,default='configs/release_version/deca_coarse.yml', help='cfg file path')
#     parser.add_argument('--mode', type=str, help='deca mode')
#     # parser.add_argument('--mode', type=str, default = 'train', help='deca mode')
#
#     args = parser.parse_args()
#     print(args, end='\n\n')
#
#     cfg = get_cfg_defaults()
#     cfg.cfg_file = None
#     cfg.mode = args.mode
#     # import ipdb; ipdb.set_trace()
#     if args.cfg is not None:
#         cfg_file = args.cfg
#         cfg = update_cfg(cfg, args.cfg)
#         cfg.cfg_file = cfg_file
#
#     return cfg
