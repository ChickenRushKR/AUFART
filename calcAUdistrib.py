
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import cv2
from loguru import logger
from datetime import datetime
from tqdm import tqdm

from decalib.utils import util
from decalib.utils.config import cfg
from decalib.utils import lossfunc2 as lossfunc
from decalib.models.expression_loss import ExpressionLossNet
from decalib.datasets import build_datasets
from decalib.models.OpenGraphAU.model.MEFL import MEFARG
from decalib.models.OpenGraphAU.utils import load_state_dict
from decalib.models.OpenGraphAU.utils import *
from decalib.models.OpenGraphAU.conf import get_config,set_logger,set_outdir,set_env

# from .datasets import build_datasets_NoAug
torch.backends.cudnn.benchmark = True
# from decalib.trainer import Trainer
from tqdm import tqdm

class calcAUdistrib(object):
    def __init__(self, config=None, device='cuda:0'):
        if config is None:
            self.cfg = cfg
        else:
            self.cfg = config
        self.device = device
        self.batch_size = 1
        self.image_size = self.cfg.dataset.image_size
        self.uv_size = self.cfg.model.uv_size
        self.K = self.cfg.dataset.K
        self.auconf = get_config()
        self.auconf.evaluate = True
        self.auconf.gpu_ids = os.environ["CUDA_VISIBLE_DEVICES"]
        set_env(self.auconf)
        self.AUNet = MEFARG(num_main_classes=self.auconf.num_main_classes, num_sub_classes=self.auconf.num_sub_classes, backbone=self.auconf.arc).to(self.device)
        self.AUNet = load_state_dict(self.AUNet, self.auconf.resume).to(self.device)
        self.AUNet.eval()

    def prepare_data(self):
        # self.train_dataset = build_datasets_NoAug.build_train(self.cfg.dataset)
        self.train_dataset = build_datasets.build_train(self.cfg.dataset)
        logger.info('---- training data numbers: ', len(self.train_dataset))

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                            num_workers=self.cfg.dataset.num_workers,
                            pin_memory=True,
                            drop_last=True)
        self.train_iter = iter(self.train_dataloader)

    def fit(self):
        self.prepare_data()

        iters_every_epoch = int(len(self.train_dataset)/self.batch_size)
        start_epoch = iters_every_epoch
        # for epoch in range(start_epoch, self.cfg.train.max_epochs):
            # random.shuffle(self.train_dataset)
            # for step, batch in enumerate(tqdm(self.train_dataloader, desc=f"Epoch: {epoch}/{self.cfg.train.max_epochs}")):
        # au_class_sum = np.zeros((1,41))
        for step in tqdm(range(iters_every_epoch)):
            try:
                batch = next(self.train_iter)
            except:
                self.train_iter = iter(self.train_dataloader)
                batch = next(self.train_iter)
            images_224 = batch['image_224'].to(self.device)[0]
            with torch.no_grad():
                au_result = self.AUNet(images_224)[1]
            seg_path = batch['seg_path'][0]
            au_path = seg_path.replace('masks','au')
            # if not os.path.exists(au_path):
            #     os.makedirs(au_path, exist_ok=True)
            au_result = torch.tensor(au_result)
            seg_path.split('/')
            np.save(au_path, au_result.cpu().numpy())
            # au_class = au >= 0.5
            # if step == 0:
                # au_class_sum[0] = au_class
            # au_class_sum[0,:] += au_class
        # au_class_sum /= range(iters_every_epoch)
        # np.save('/mnt/hdd/EncoderTrainingCode/Code/data/AU_info.npy', au_class_sum)

    def checkDistrib(self):
        auinfo = np.load('/mnt/hdd/EncoderTrainingCode/Code/data/AU_info.npy')
        print('AU info all: ', auinfo)
        numofdata = auinfo.shape[0]
        binary_data = auinfo >= 0.5
        class_freq = np.mean(binary_data, axis=0)
        non_zero = class_freq[class_freq > 0]
        class_weight = 1/(non_zero + 1e-6)
        class_weight = class_weight / np.sum(class_weight)
        zero_freq_ind = np.where(class_freq == 0)[0]
        if len(zero_freq_ind) > 0:
            avg_weight = np.mean(class_weight)
            for i in zero_freq_ind:
                class_weight = np.insert(class_weight, i, avg_weight)

        for class_idx, weight in enumerate(class_weight):
            print(f"class {class_idx}: weight = {weight:.4f}")

        scaled_weight = (class_weight - min(class_weight)) / (max(class_weight))
        mean = 0.5; std_dev = 0.1
        np.random.seed(0)
        gaussian_weight = np.random.normal(mean, std_dev, len(scaled_weight))

        gaussian_weight = gaussian_weight[np.argsort(class_weight)]

        for class_idx, weight in enumerate(gaussian_weight):
            print(f"class {class_idx}: Gauss_weight = {weight:.4f}")

        print(np.save('/mnt/hdd/EncoderTrainingCode/Code/data/AU_weight.npy', gaussian_weight))



def main(cfg):
    cfg.rasterizer_type = 'pytorch3d'
    cfg.device = 'cuda:1'
    AUdistrib = calcAUdistrib(config=cfg)

    # AUdistrib.checkDistrib()
    AUdistrib.fit()

# increase weighted landmarks loss of mouth
if __name__ == '__main__':
    from decalib.utils.config import parse_args

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    cfg = parse_args()
    main(cfg)