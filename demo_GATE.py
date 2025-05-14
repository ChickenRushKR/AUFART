import os, sys
import cv2
import numpy as np
import argparse
from tqdm import tqdm
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from aufartlib.gatfarec import aufart
from aufartlib.datasets import datasets
from aufartlib.utils.config import cfg as aufart_cfg
from datetime import datetime
import time

def main(args):
    savefolder = args.savefolder
    device = args.device
    os.makedirs(savefolder, exist_ok=True)

    aufart_cfg.model.use_tex = args.useTex
    aufart_cfg.rasterizer_type = args.rasterizer_type
    aufart_cfg.model.extract_tex = args.extractTex
    aufart_cfg.pretrained_modelpath = args.pretrained_modelpath
    aufart_cfg.model.flame_model_path = '/home/cine/LGAI/Final_Code/TestPart/data/generic_model.pkl'

    if not os.path.exists(args.pretrained_modelpath): exit()

    aufart = aufart(config=aufart_cfg, device=args.device)
    os.makedirs(os.path.join(savefolder, 'result'), exist_ok=True)
    os.makedirs(os.path.join(savefolder, 'result_original'), exist_ok=True)

    testdata = datasets.TestData(args.inputpath, iscrop=args.iscrop, crop_size=aufart_cfg.dataset.image_size, scale=1.05,
                                 face_detector=args.detector, ifCenter='', ifSize='')

    if args.shapeaggr:
        shape_img = []
        for i in range(0, len(testdata), len(testdata) // 20):
            data = testdata[i]
            images_224 = data['image_224'].to(device)[None, ...]
            shape_img.append(images_224)
        shape_img = torch.stack(shape_img, 1).to(device).squeeze(0)
        newshape = shape_aggr.forward(images=shape_img)

    time_per_frame = 0.0
    print(datetime.now())
    for i in tqdm(range(len(testdata))):
        starttime = time.time()
        data = testdata[i]
        name = data['imagename']
        images_224 = data['image_224'].to(device)[None, ...]

        with torch.no_grad():
            codedict = aufart.encode(images_224, use_detail=args.useDetail)
            if args.shapeaggr:
                codedict['shape'] = newshape
            opdict, visdict = aufart.decode(codedict, use_detail=args.useDetail)
            if args.render_orig:
                tform = data['tform'][None, ...]
                tform = torch.inverse(tform).transpose(1, 2).to(device)
                original_image = data['original_image'][None, ...].to(device)
                orig_opdict, orig_visdict = aufart.decode(codedict, render_orig=True, original_image=original_image, tform=tform, use_detail=args.useDetail)
                orig_visdict['inputs'] = original_image
        endtime = time.time()
        time_per_frame += (endtime - starttime)

        vis_image = aufart.visualize(visdict, size=448)
        cv2.imwrite(os.path.join(savefolder, 'result', name + '_vis.jpg'), vis_image)
        if args.render_orig:
            vis_image2 = aufart.visualize(orig_visdict, size=448)
            cv2.imwrite(os.path.join(savefolder, 'result_original', name + '_vis.jpg'), vis_image2)

    print(f'-- please check the results in {savefolder}')
    time_per_frame /= len(testdata)
    print(f'--Average time per frame is {time_per_frame}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='aufart: Detailed Expression Capture and Animation')
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    parser.add_argument('-i', '--inputpath', default='/mnt/hdd/dataset/300W/', type=str,
                        help='path to the test data, can be image folder, image path, image list, video')
    parser.add_argument('-s', '--savefolder', default='videoResult/testGATE24/', type=str,
                        help='path to the output directory, where results(obj, txt files) will be stored.')
    parser.add_argument('--pretrained_modelpath', default='/mnt/hdd/EncoderTrainingCode/Code/Training/testGATE24/model.tar', type=str, help='model.tar path')
    parser.add_argument('--shapeaggr', default=False, type=bool, help='shape aggregation')
    parser.add_argument('--useDetail', default=False, type=lambda x: x.lower() in ['true', '1'], help='use_detail')
    parser.add_argument('--device', default='cuda:1', type=str,
                        help='set device, cpu for using cpu')
    parser.add_argument('--iscrop', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to crop input image, set false only when the test image are well cropped')
    parser.add_argument('--detector', default='retinaface', type=str,
                        help='detector for cropping face, check aufartlib/detectors.py for details')
    parser.add_argument('--rasterizer_type', default='pytorch3d', type=str,
                        help='rasterizer type: pytorch3d or standard')
    parser.add_argument('--render_orig', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to render results in original image size, currently only works when rasterizer_type=standard')
    parser.add_argument('--useTex', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to use FLAME texture model to generate uv texture map, \
                            set it to True only if you downloaded texture model')
    parser.add_argument('--extractTex', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to extract texture from input image as the uv texture map, set false if you want albeo map from FLAME mode')
    main(parser.parse_args())
