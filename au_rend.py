import os, sys
import cv2
import numpy as np
import argparse
from tqdm import tqdm
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decalib.utils import util
from decalib.gatfarec import DECA
# from decalib.models.cnet import Shape_aggregation
from decalib.datasets import datasets
from decalib.utils.config import cfg as deca_cfg
from datetime import datetime
from decalib.models.OpenGraphAU.model.MEFL import MEFARG
from decalib.models.OpenGraphAU.utils import load_state_dict
from decalib.models.OpenGraphAU.utils import *
from decalib.models.OpenGraphAU.conf import get_config,set_logger,set_outdir,set_env
import copy
import time
def main(args):
    # if args.rasterizer_type != 'standard':
    #     args.render_orig = False
    savefolder = args.savefolder
    device = args.device
    os.makedirs(savefolder, exist_ok=True)

    # run DECA
    deca_cfg.model.use_tex = args.useTex
    deca_cfg.rasterizer_type = args.rasterizer_type
    deca_cfg.model.extract_tex = args.extractTex
    deca_cfg.pretrained_modelpath = args.pretrained_modelpath
    
    deca_cfg.model.flame_model_path = '/home/cine/LGAI/Final_Code/TestPart/data/generic_model.pkl'
    
    if not os.path.exists(args.pretrained_modelpath): exit()
 
    deca = DECA(config = deca_cfg, device=args.device)

    AU_net = MEFARG(num_main_classes=27, num_sub_classes=14, backbone='resnet').to(device)
    AU_net = load_state_dict(AU_net, '/mnt/hdd/EncoderTrainingCode/Code/decalib/models/OpenGraphAU/checkpoints/OpenGprahAU-ResNet50_second_stage.pth').to(device)
    AU_net.eval()
    #
    os.makedirs(os.path.join(savefolder, 'result'), exist_ok=True)
    os.makedirs(os.path.join(savefolder, 'result_original'), exist_ok=True)

    os.makedirs(os.path.join(savefolder, 'obj'), exist_ok=True)
    # os.makedirs(os.path.join(savefolder, 'npy'), exist_ok=True)
    
    # load test images
    testdata = datasets.TestData(args.inputpath, iscrop=args.iscrop,crop_size=deca_cfg.dataset.image_size, scale=1.05,
                                 face_detector=args.detector, ifCenter='',ifSize='' )

    if args.shapeaggr:
        shape_img = []
        for i in range(0, len(testdata), len(testdata) // 20):
            data = testdata[i]
            images_224 = data['image_224'].to(device)[None,...]
            shape_img.append(images_224)
        shape_img = torch.stack(shape_img,1).to(device).squeeze(0)
        newshape = shape_aggr.forward(images=shape_img)
    # uv_texture_gt = torch.tensor(np.load(os.path.join(savefolder, 'uv_texture_gt.npy'))).to(device)
    time_per_frame = 0.0
    b = 0.95
    print(datetime.now())
    for i in tqdm(range(len(testdata))):
        starttime = time.time()
        data = testdata[i]
        name = data['imagename']
        images_224 = data['image_224'].to(device)[None,...]

        with torch.no_grad():
            codedict  = deca.encode(images_224, use_detail=True)
            
            # codedict, _  = deca.encode(images, images_224, run_224= True)
            if args.shapeaggr:
                codedict['shape'] = newshape
            # codedict['uv_texture_gt'] = uv_texture_gt
            # codedict['shape'][:,:] = 0.0
            # codedict['shape'] = shape
            codedict['tex'][:,:] *= 3.0
            codedict['light'][:,:] = 0.0
            opdict, visdict = deca.decode(codedict, use_detail=False) #tensor
            predAU = AU_net(opdict['rendered_images'])[1]
            gtAU = AU_net(images_224)[1]
            visdict['au_gt'] = util.draw_activation_circles(images_224, opdict['landmarks2d'], gtAU)
            visdict['au_rend'] = util.draw_activation_circles(opdict['rendered_images'], opdict['landmarks2d'], predAU)
            # print(predAU)
            # print(codedict['tex'])
            # print(codedict['light'])
            #     codedict['cam'] = b* previous + codedict['cam']*(1-b)
            # previous = codedict['cam']
            endtime = time.time()
            if args.render_orig:
                tform = data['tform'][None, ...]
                tform = torch.inverse(tform).transpose(1,2).to(device)
                original_image = data['original_image'][None, ...].to(device)
                orig_opdict, orig_visdict = deca.decode(codedict, render_orig=True, original_image=original_image, tform=tform, use_detail=False)
                orig_visdict['inputs'] = original_image
        time_per_frame += (endtime - starttime)
        # #
        # codedict_neut = copy.deepcopy(codedict); codedict_neut['pose'][:,:3] = 0.0; codedict_neut['shape'][:,:] = 0.0;
        # opdict_neut, _ = deca.decode(codedict_neut) #tensor
        # if i !=0:
        # deca.save_obj(os.path.join(savefolder, 'obj', name + '.obj'), opdict_neut)
        # for m in codedict:
        #     codedict[m] = codedict[m].to('cpu')
        # np.save(os.path.join(savefolder, 'npy', name + '.npy'), codedict)
        vis_image = deca.visualize(visdict, size=448) # 'size' is the visualized image size. 
        cv2.imwrite(os.path.join(savefolder, 'result', name + '_vis.jpg'), vis_image)
        if args.render_orig:
            vis_image2 = deca.visualize(orig_visdict, size=448)
            cv2.imwrite(os.path.join(savefolder, 'result_original', name + '_vis.jpg'), vis_image2)
        # break
        #
        # np.save(os.path.join(savefolder, 'uv_texture_gt.npy'), opdict['uv_texture_gt'].to('cpu').numpy())
        # cv2.imwrite(os.path.join(savefolder, 'uv_texture_gt.png')s, opdict['uv_texture_gt'].to('cpu').numpy())
        
        # out.write(vis_image)
        # out1.write(vis_image2)

    print(f'-- please check the results in {savefolder}')
    time_per_frame /= len(testdata)
    print(f'--Average time per frame is {time_per_frame}')
    # out.release()
    # out1.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DECA: Detailed Expression Capture and Animation')
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
    # parser.add_argument('-i', '--inputpath', default='/home/cine/emoca/data/EMOCA_test_example_data/videos/82-25-854x480_affwild2/', type=str,
    parser.add_argument('-i', '--inputpath', default='/mnt/hdd/CINeLabDataset2/Actor03/angry/images/', type=str,
    # parser.add_argument('-i', '--inputpath', default='/mnt/hdd/Downloads/woman_SR_sequence/1_1/', type=str,
                        help='path to the test data, can be image folder, image path, image list, video')
    parser.add_argument('-s', '--savefolder', default='videoResult/testGATE3_auvis/', type=str,
                        help='path to the output directory, where results(obj, txt files) will be stored.')
    # parser.add_argument('--videoName', default='resultWithoutTC.avi', type=str,)
    parser.add_argument('--pretrained_modelpath', default='/mnt/hdd/EncoderTrainingCode/Code/Training/testGATE3/model.tar', type=str, help='model.tar path')
    parser.add_argument('--shapeaggr', default=False, type=bool, help='shape aggregation')
    parser.add_argument('--device', default='cuda:1', type=str,
                        help='set device, cpu for using cpu' )
    # process test images
    parser.add_argument('--iscrop', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to crop input image, set false only when the test image are well cropped' )
 
    parser.add_argument('--detector', default='retinaface', type=str, # fan or retinaface
                        help='detector for cropping face, check decalib/detectors.py for details' )
    # rendering option
    parser.add_argument('--rasterizer_type', default='pytorch3d', type=str,
                        help='rasterizer type: pytorch3d or standard' )
    parser.add_argument('--render_orig', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to render results in original image size, currently only works when rasterizer_type=standard')
    # save
    parser.add_argument('--useTex', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to use FLAME texture model to generate uv texture map, \
                            set it to True only if you downloaded texture model' )
    parser.add_argument('--extractTex', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to extract texture from input image as the uv texture map, set false if you want albeo map from FLAME mode' ) 
    main(parser.parse_args())