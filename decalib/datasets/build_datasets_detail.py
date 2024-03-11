import os, sys
import random
import cv2
import numpy as np
from skimage.io import imread
from skimage.transform import estimate_transform, warp, resize, rescale
import torch
from torch.utils.data import Dataset, ConcatDataset
from glob import glob


def build_train(config, is_train=True):
    data_list = []
    data_list.append(SelfDataset(K=config.K,  image_size=config.image_size , mediapipePath = config.mediapipePath))
    dataset = ConcatDataset(data_list)
    return dataset


class SelfDataset(Dataset):
    def __init__(self, K,  image_size, mediapipePath = 'mediapipe_landmark_embedding.npz'):
        '''
        K must be less than 6
        '''
        self.mediapipe_idx = \
            np.load(mediapipePath,
                    allow_pickle=True, encoding='latin1')[
                'landmark_indices'].astype(int)
        self.K = K
        # allName = ['FFHQ', 'FFHQ-Aug', 'CelebAHQ', 'CelebAHQ-Aug']
        self.image_size = image_size

        self.source = ['/mnt/hdd/dataset/BUPT/masks', '/mnt/hdd/dataset/FFHQ/masks', '/mnt/hdd/dataset/CelebAHQ/masks']
        # self.source = ['/mnt/hdd/dataset/Affwild/masks']

        self.allmasksFolder = (glob(self.source[0] + '/*.npy') + glob(self.source[1] + '/*.npy') + glob(self.source[2] + '/*.npy'))

        # self.source = ['/media/cine/First/Old/FFHQ/masks/',
        #                '/media/cine/First/Old/FFHQ_Aug/masks/',
        #                '/media/cine/First/Old/CelebAHQ/masks/',
        #                '/media/cine/First/Old/CelebAHQ_Aug/masks/'
        #                ]
        #
        # self.allmasksFolder = (
        #             glob(self.source[0] + '/*.npy') + glob(self.source[1] + '/*.npy')+
        #             glob(self.source[2] + '/*.npy')+ glob(self.source[3] + '/*.npy') )

        random.shuffle(self.allmasksFolder)

    def shuffle(self):
        random.shuffle(self.allmasksFolder)

    def __len__(self):
        return len(self.allmasksFolder)

    def __getitem__(self, idx):
        images_224_lists = [];
        images_list = [];
        kpt_list = [];
        dense_kpt_list = [];
        mask_list = []

        name = os.path.splitext(os.path.split(self.allmasksFolder[idx])[-1])[0]
        # name = os.path.splitext(os.path.split(self.allkptFiles[idx])[-1])[0]
        seg_path = self.allmasksFolder[idx]
        if os.path.exists(seg_path.replace('masks', 'images').replace('.npy', '.jpg')):
            image_path = seg_path.replace('masks', 'images').replace('.npy', '.jpg')
        else:
            image_path = seg_path.replace('masks', 'images').replace('.npy', '.png')
        kpt_path = seg_path.replace('masks','kpts')
        kpt_path_mp = seg_path.replace('masks', 'kpts_dense')

        # dense_lmks =  np.load(kpt_path_mp)[:, :2]
        lmks = np.load(kpt_path)
        dense_lmks = np.load(kpt_path_mp)
        image = imread(image_path) / 255. 
        mask = self.load_mask(seg_path, image.shape[0], image.shape[1])
        mask = cv2.resize(mask, (224, 224))

        images_224_lists.append(cv2.resize(image, (224, 224)).transpose(2, 0, 1))
        # images_list.append(image.transpose(2, 0, 1))
        kpt_list.append(lmks)
        dense_kpt_list.append(dense_lmks[self.mediapipe_idx, :])
        mask_list.append(mask)

        images_224_array = torch.from_numpy(np.array(images_224_lists)).type(dtype=torch.float32)
        # images_array = torch.from_numpy(np.array(images_list)).type(dtype=torch.float32)
        kpt_array = torch.from_numpy(np.array(kpt_list)).type(dtype=torch.float32)
        dense_kpt_array = torch.from_numpy(np.array(dense_kpt_list)).type(dtype=torch.float32)
        mask_array = torch.from_numpy(np.array(mask_list)).type(dtype=torch.float32)

        data_dict = {
            'image_224': images_224_array,
            # 'image': images_array,
            'landmark': kpt_array,
            'landmark_dense': dense_kpt_array,
            'mask': mask_array
        }
        return data_dict


    def load_mask(self, maskpath, h, w):
        if os.path.isfile(maskpath):
            vis_parsing_anno = np.load(maskpath)

            # mask = np.zeros_like(vis_parsing_anno)
            mask = np.zeros((h, w))

            # index = vis_parsing_anno > 0
            mask[vis_parsing_anno > 0] = 1.
            mask[vis_parsing_anno == 2] = 1.
            mask[vis_parsing_anno == 3] = 1.
            mask[vis_parsing_anno == 4] = 1.
            mask[vis_parsing_anno == 5] = 1.
            mask[vis_parsing_anno == 9] = 1.
            mask[vis_parsing_anno == 7] = 1.
            mask[vis_parsing_anno == 8] = 1.
            mask[vis_parsing_anno == 10] = 0  # hair
            mask[vis_parsing_anno == 11] = 0  # left ear
            mask[vis_parsing_anno == 12] = 0  # right ear
            mask[vis_parsing_anno == 13] = 0  # glasses
            # print('shape...',mask.shape)
        else:
            mask = np.ones((h, w, 3))
        return mask


