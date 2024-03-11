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
from sort.sort import *
import numpy as np
class FAN(object):
    def __init__(self, type):
        import face_alignment
        self.model = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
        # self.model = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False,face_detector = 'blazeface' )
        self.preKpt = []
        self.scale = 1.6
        self.type = type
        self.mot_tracker = Sort()
    
    def lmk(self, image):
        out = self.model.get_landmarks(image)

    def run(self,image):
        '''
        image: 0-255, uint8, rgb, [h, w, 3]
        return: detected box list
        '''
        out = self.model.get_landmarks(image)
        h,w,_= image.shape
        lenMax = max(h,w)
        bboxes = []
        if out is None:
            return [0], 'kpt68'
        else:
            if len(self.preKpt)==0 or len(out)==1 or self.type=='image':
                kpt = out[0].squeeze()
                left = np.min(kpt[:, 0]);
                right = np.max(kpt[:, 0]);
                top = np.min(kpt[:, 1]);
                bottom = np.max(kpt[:, 1])
                bbox = [left, top, right, bottom]
            else:
                for o in out:
                    kpt = o.squeeze()

                    left = np.min(kpt[:, 0]);
                    right = np.max(kpt[:, 0]);
                    top = np.min(kpt[:, 1]);
                    bottom = np.max(kpt[:, 1])
                    bboxes.append([left,top, right, bottom])
                track_bbs_ids = self.mot_tracker.update(np.array(bboxes))  # !!!
                bbox = track_bbs_ids[-1, :4]

            # self.preKpt = []
            # self.preKpt.extend(bbox)
            return bbox, 'kpt68'

class RetinaFace:
    def __init__(self, type):
        from retinaface import RetinaFace
        self.face_detector = RetinaFace()
        self.type = type
        self.mot_tracker = Sort()

    def run(self, image):
        obj = self.face_detector.predict(rgb_image=image, threshold=0.9)
        if obj is None or len(obj)==0:
            return [0], 'bbox'
        identity = obj[0]

        left = identity['x1']
        right = identity['x2']
        top = identity['y1']
        bottom = identity['y2']
        bbox = [left, top, right, bottom]

        if self.type =='video':
            bboxes = []
            for identity in obj:
                bboxes.append([identity['x1'], identity['y1'],identity['x2'], identity['y2']])
            track_bbs_ids = self.mot_tracker.update(np.array(bboxes))  # !!!
            bbox = track_bbs_ids[0, :4]
            # bbox = track_bbs_ids[-1, :4]
            # bbox = track_bbs_ids[0, :4]
        return bbox, 'bbox'

