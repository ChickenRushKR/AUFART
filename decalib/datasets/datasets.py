# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at deca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

import os
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
from skimage.io import imread
from skimage.transform import estimate_transform, warp, resize, rescale
import glob
from . import detectors_orig as detectors

def video2sequence(video_path ):
    videofolder = os.path.splitext(video_path)[0]
    os.makedirs(videofolder, exist_ok=True)
    video_name = os.path.splitext(os.path.split(video_path)[-1])[0]
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 1
    imagepath_list = []
    while success: 
        imagepath = os.path.join(videofolder, f'{video_name}_frame{count:04d}.jpg')
        cv2.imwrite(imagepath, image)  # save frame as JPEG file
        success, image = vidcap.read()
        count += 1
        imagepath_list.append(imagepath)
    print('video frames are stored in {}'.format(videofolder))
    return imagepath_list


class TestData(Dataset):
    def __init__(self, testpath, iscrop=True, crop_size=448, scale=1.25, face_detector='fan',  
                 ifCenter='', ifSize=''):
        '''
            testpath: folder, imagepath_list, image path, video path
        '''
        if isinstance(testpath, list):
            self.imagepath_list = testpath
            self.type = 'image'
        elif os.path.isdir(testpath):
            self.imagepath_list = glob.glob(testpath + '/*.jpg') + glob.glob(testpath + '/*.png') + glob.glob(
                testpath + '/*.bmp') + glob.glob(testpath + '/*.jpeg') + glob.glob(testpath + '/*/*/*.jpg')
            self.type = 'image'
        elif os.path.isfile(testpath) and (testpath[-3:] in ['jpg', 'png', 'bmp', 'jpeg']):
            self.imagepath_list = [testpath]
            self.type = 'image'
        elif os.path.isfile(testpath) and (testpath[-3:] in ['mp4', 'csv', 'vid', 'ebm', 'avi']):
            self.imagepath_list = video2sequence(testpath)
            self.type = 'video'
        else:
            print(f'please check the test path: {testpath}')
            exit()
        try:
            self.imagepath_list = sorted(self.imagepath_list,
                                         key=lambda x: int(os.path.splitext(os.path.split(x)[-1])[0]))
        except:
            self.imagepath_list = sorted(self.imagepath_list)

        self.crop_size = crop_size
        self.scale = scale
        self.iscrop = iscrop
        self.resolution_inp = crop_size
        self.detector_type = face_detector
        self.scene_size = crop_size
        if face_detector == 'fan':
            self.face_detector = detectors.FAN(self.type)
        elif face_detector == 'retinaface':
            self.face_detector = detectors.RetinaFace(self.type)
        elif face_detector == 'retina_fan':
            self.face_detector = detectors.RetinaFace(self.type)
            self.face_detector_fan = detectors.FAN(self.type)
        # elif face_detector == 'mtcnn':
        #     self.face_detector = detectors.MTCNN()
        else:
            print(f'please check the detector: {face_detector}')
            exit()

        self.ifCenter = ifCenter
        self.ifSize = ifSize
        self.preCenter = []
        self.preSize = 0
        self.preSizeA = []
        self.b = 0.95

    def __len__(self):
        return len(self.imagepath_list)

    def bbox2point(self, left, right, top, bottom, type='bbox'):
        ''' bbox from detector and landmarks are different
        '''
        if type == 'kpt68':
            old_size = (right - left + bottom - top) / 2 * 1.1
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
        elif type == 'bbox':
            old_size = (right - left + bottom - top) / 2
            # old_size = max((right - left),( bottom - top))
            # center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 + old_size * 0.12])
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0  + old_size*0.15])
        else:
            raise NotImplementedError

        if self.ifCenter == 'EMA':
            if len(self.preCenter) != 0:
                center = self.b * self.preCenter + center * (1 - self.b)
            self.preCenter = center
        elif self.ifCenter == 'SMA':
            self.preCenter.append(center)
            if len(self.preCenter) >= 3:
                center = (self.preCenter[0] + self.preCenter[1] + self.preCenter[2]) / 3
                del self.preCenter[0]
        return old_size, center

    def __getitem__(self, index):
        imagepath = self.imagepath_list[index]
        imagename = os.path.splitext(os.path.split(imagepath)[-1])[0]
        image = np.array(imread(imagepath))
        
        scene_image = image / 255.
        # random crop scene
        square_scene = self.scene_crop(scene_image)

        if len(image.shape) == 2:
            image = image[:, :, None].repeat(1, 1, 3)
        if len(image.shape) == 3 and image.shape[2] > 3:
            image = image[:, :, :3]

        h, w, _ = image.shape
        if self.iscrop:

            bbox, bbox_type = self.face_detector.run(image)

            if len(bbox) < 4:
                print('no face detected! run original image!!')
                left = 0; right = h - 1; top = 0; bottom = w - 1
            else:
                left = bbox[0];  right = bbox[2]; top = bbox[1]; bottom = bbox[3]
            old_size, center = self.bbox2point(left, right, top, bottom, type=bbox_type)

            size = int(old_size * self.scale)

            if self.ifCenter == 'EMA':
                if self.preSize != 0:
                    size = self.b * self.preSize + size * (1 - self.b)
                self.preSize = size

            elif self.ifCenter == 'SMA':
                if self.ifSize:
                    self.preSizeA.append(size)
                    if len(self.preSizeA) >= 3:
                        size = (self.preSizeA[0] + self.preSizeA[1] + self.preSizeA[2]) / 3
                        del self.preSizeA[0]
            src_pts = np.array(
                [[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                 [center[0] + size / 2, center[1] - size / 2]])
        else:
            src_pts = np.array([[0, 0], [0, h - 1], [w - 1, 0]])

        DST_PTS = np.array([[0, 0], [0, self.resolution_inp - 1], [self.resolution_inp - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)

        image = image / 255.

        dst_image = warp(image, tform.inverse, output_shape=(self.resolution_inp, self.resolution_inp))
        dst_image = dst_image.transpose(2, 0, 1)

        # if self.detector_type == 'retina_fan':
        #     image_255 = image * 255.
        #     bbox, bbox_type = self.face_detector_fan.run(image_255)

        #     if len(bbox) < 4:
        #         print('no face detected! run original image!!')
        #         left = 0; right = h - 1; top = 0; bottom = w - 1
        #     else:
        #         left = bbox[0];  right = bbox[2]; top = bbox[1]; bottom = bbox[3]
        #     old_size, center = self.bbox2point(left, right, top, bottom, type=bbox_type)

        #     size = int(old_size * self.scale)
        #     src_pts = np.array(
        #         [[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
        #          [center[0] + size / 2, center[1] - size / 2]])
        DST_PTS = np.array([[0, 0], [0, 224 - 1], [224 - 1, 0]])
        tform_224 = estimate_transform('similarity', src_pts, DST_PTS)

        dst_image_224 = warp(image, tform_224.inverse, output_shape=(224, 224))
        dst_image_224 = dst_image_224.transpose(2, 0, 1)
        scene_array = torch.from_numpy(square_scene.transpose(2, 0, 1)).type(dtype=torch.float32)

        return {'image_224': torch.tensor(dst_image_224).float(),
                'image': torch.tensor(dst_image).float(),
                'scene_image': scene_array,
                'imagename': imagename,
                'tform': torch.tensor(tform.params).float(),
                'tform_224': torch.tensor(tform_224.params).float(),
                'original_image': torch.tensor(image.transpose(2, 0, 1)).float(),
                'imagepath': imagepath,
                }

    def scene_crop(self, image):
        scene_h, scene_w = image.shape[:2]
        if scene_w > scene_h:
            sq_size = scene_h
            random_left = np.random.randint(scene_w - sq_size)
            square_scene = image[0:sq_size, random_left:random_left + sq_size]
            square_scene = cv2.resize(square_scene, (self.scene_size, self.scene_size), interpolation=cv2.INTER_AREA)
        elif scene_h > scene_w:
            sq_size = scene_w
            random_top = np.random.randint(scene_h - sq_size)
            square_scene = image[random_top: random_top+sq_size, 0:sq_size]
            square_scene = cv2.resize(square_scene, (self.scene_size, self.scene_size), interpolation=cv2.INTER_AREA)
        else:
            square_scene = cv2.resize(image, (self.scene_size, self.scene_size), interpolation=cv2.INTER_AREA)
        return square_scene