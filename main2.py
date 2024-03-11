''' training script of DECA
'''
import os, sys
import numpy as np
import yaml
import torch
import torch.backends.cudnn as cudnn
import torch
import shutil
from copy import deepcopy

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
np.random.seed(0)

def main(cfg):
    # creat folders
    os.makedirs(os.path.join(cfg.output_dir, cfg.train.log_dir), exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, cfg.train.vis_dir), exist_ok=True)
    # os.makedirs(os.path.join(cfg.output_dir, cfg.train.val_vis_dir), exist_ok=True)
    with open(os.path.join(cfg.output_dir, cfg.train.log_dir, 'full_config.yaml'), 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)
    shutil.copy(cfg.cfg_file, os.path.join(cfg.output_dir, 'config.yaml'))

    # cudnn related setting
    cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True

    # start training
    # deca model
    # from decalib.mymodel import mymodel
    from decalib.gatfarec2 import DECA
    from decalib.trainer2 import Trainer
    cfg.rasterizer_type = 'pytorch3d'
    cfg.device = 'cuda:1'
    mymodel = DECA(cfg)
    trainer = Trainer(model=mymodel, config=cfg)

    ## start train
    trainer.fit()

# increase weighted landmarks loss of mouth
if __name__ == '__main__':
    from decalib.utils.config import parse_args

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    cfg = parse_args()
    if cfg.cfg_file is not None: 
        exp_name = cfg.cfg_file.split('/')[-1].split('.')[0]
        cfg.exp_name = exp_name
    main(cfg)

# run:
# python main_train.py --cfg configs/release_version/deca_pretrain.yml
