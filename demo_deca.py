import os, sys
import cv2
import numpy as np
import argparse
from tqdm import tqdm
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decalib.deca import DECA
# from decalib.models.cnet import Shape_aggregation
from decalib.datasets import datasets
from decalib.utils.config import cfg as deca_cfg
from decalib.utils.util import vis_au
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

    auconf = get_config()
    auconf.evaluate = True
    auconf.gpu_ids = os.environ["CUDA_VISIBLE_DEVICES"]
    set_env(auconf)
    AU_net = MEFARG(num_main_classes=auconf.num_main_classes, num_sub_classes=auconf.num_sub_classes, backbone=auconf.arc).to(device)
    AU_net = load_state_dict(AU_net, auconf.resume).to(device)
    AU_net.eval()

    # run DECA
    deca_cfg.model.use_tex = args.useTex
    deca_cfg.rasterizer_type = args.rasterizer_type
    deca_cfg.model.extract_tex = args.extractTex
    deca_cfg.pretrained_modelpath = args.pretrained_modelpath
    deca_cfg.model.tex_type = 'BFM'
    deca_cfg.model.n_tex = 50
    
    deca_cfg.model.flame_model_path = '/home/cine/LGAI/Final_Code/TestPart/data/generic_model.pkl'
    
    if not os.path.exists(args.pretrained_modelpath): exit()
 
    deca = DECA(config = deca_cfg, device=args.device)
    # shape_aggr = Shape_aggregation(device=device, cfg=deca_cfg)
    # deca = DECA(config = deca_cfg, device='cuda:1')
    # shape = np.load("/media/cine/First/manshape.npy", allow_pickle=True)
    # # print(shape)
    # # shape = torch.tensor(shape).float().to(device)
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter(savefolder+args.videoName, fourcc, 24, (448*3,448), True)
    # out1 = cv2.VideoWriter(savefolder+"original.avi", fourcc, 24, (2388,448), True)
    #
    os.makedirs(os.path.join(savefolder, 'result'), exist_ok=True)
    os.makedirs(os.path.join(savefolder, 'result_original'), exist_ok=True)

    # os.makedirs(os.path.join(savefolder, 'obj'), exist_ok=True)
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
    csvfile = open(os.path.join(savefolder,'au.csv'), 'w')
    csvfile.write('gt0,gt1,gt2,gt3,gt4,gt5,gt6,gt7,gt8,gt9,gt10,gt11,pd0,pd1,pd2,pd3,pd4,pd5,pd6,pd7,pd8,pd9,pd10,pd11\n')
    print(datetime.now())
    for i in tqdm(range(len(testdata))):
        starttime = time.time()
        data = testdata[i]
        name = data['imagename']
        images_224 = data['image_224'].to(device)[None,...]

        with torch.no_grad():
            codedict  = deca.encode(images_224, use_detail=args.useDetail)
            # print(i, codedict['exp'])
            # codedict, _  = deca.encode(images, images_224, run_224= True)
            if args.shapeaggr:
                codedict['shape'] = newshape
            # codedict['uv_texture_gt'] = uv_texture_gt
            # codedict['shape'][:,:] = 0.0
            # codedict['shape'] = shape

            opdict, visdict = deca.decode(codedict, use_detail=args.useDetail) #tensor

            compare = [0,1,2,3,5,6,7,9,12,17,19,22]
            image_au = AU_net(images_224)[1][:,compare]
            rend_au = AU_net(opdict['rendered_images'])[1][:,compare]
            for j in range(len(compare) * 2):
                if j < len(compare):
                    # csvfile.write(f'{1 if image_au[j].item()>=0.5 else 0},')
                    csvfile.write(f'{image_au[0,j].item()},')
                else:
                    # csvfile.write(f'{1 if rend_au[j-27].item()>=0.5 else 0},')
                    csvfile.write(f'{rend_au[0,j-len(compare)].item()},')
            csvfile.write('\n')

            #     codedict['cam'] = b* previous + codedict['cam']*(1-b)
            # previous = codedict['cam']
            endtime = time.time()
            if args.render_orig:
                tform = data['tform'][None, ...]
                tform = torch.inverse(tform).transpose(1,2).to(device)
                original_image = data['original_image'][None, ...].to(device)
                orig_opdict, orig_visdict = deca.decode(codedict, render_orig=True, original_image=original_image, tform=tform, use_detail=args.useDetail)
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
        visdict['vis_au'] = vis_au(image_au,rend_au)
        vis_image = deca.visualize(visdict, size=448) # 'size' is the visualized image size. 
        cv2.imwrite(os.path.join(savefolder, 'result', name + '_vis.jpg'), vis_image)
        if args.render_orig:
            vis_image2 = deca.visualize(orig_visdict, size=448)
            cv2.imwrite(os.path.join(savefolder, 'result_original', name + '_vis.jpg'), vis_image2)
        #
        # np.save(os.path.join(savefolder, 'uv_texture_gt.npy'), opdict['uv_texture_gt'].to('cpu').numpy())
        # cv2.imwrite(os.path.join(savefolder, 'uv_texture_gt.png')s, opdict['uv_texture_gt'].to('cpu').numpy())
        
        # out.write(vis_image)
        # out1.write(vis_image2)

    csvfile.close()
    print(f'-- please check the results in {savefolder}')
    time_per_frame /= len(testdata)
    print(f'--Average time per frame is {time_per_frame}')
    # out.release()
    # out1.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DECA: Detailed Expression Capture and Animation')
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    # parser.add_argument('-i', '--inputpath', default='/home/cine/emoca/data/EMOCA_test_example_data/videos/82-25-854x480_affwild2/', type=str,
    # parser.add_argument('-i', '--inputpath', default='/home/cine/Downloads/disfa_1_left/', type=str,
    parser.add_argument('-i', '--inputpath', default='/mnt/hdd/dataset/DISFA/LeftVideoSN001_comp_sequence/1_1/', type=str,
    # parser.add_argument('-i', '--inputpath', default='/mnt/hdd/dataset/now/multiview_expressions/', type=str,
    # parser.add_argument('-i', '--inputpath', default='/mnt/hdd/CINeLabDataset2/Actor18/sad/images/', type=str,
    # parser.add_argument('-i', '--inputpath', default='/mnt/hdd/ravdess_image/', type=str,
    # parser.add_argument('-i', '--inputpath', default='/mnt/hdd/dataset/300W/', type=str,
    # parser.add_argument('-i', '--inputpath', default='/mnt/hdd/Downloads/woman_SR_sequence/1_1/', type=str,
                        help='path to the test data, can be image folder, image path, image list, video')
    parser.add_argument('-s', '--savefolder', default='videoResult/deca_DISFA01_detail/', type=str,
                        help='path to the output directory, where results(obj, txt files) will be stored.')
    # parser.add_argument('--videoName', default='resultWithoutTC.avi', type=str,)
    parser.add_argument('--pretrained_modelpath', default='/mnt/hdd/EncoderTrainingCode/Code/Training/testGATE22/model.tar', type=str, help='model.tar path')
    parser.add_argument('--shapeaggr', default=False, type=bool, help='shape aggregation')
    parser.add_argument('--useDetail', default=True, type=lambda x: x.lower() in ['true', '1'], help='use_detail')
    parser.add_argument('--device', default='cuda:1', type=str,
                        help='set device, cpu for using cpu' )
    # process test images
    parser.add_argument('--iscrop', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to crop input image, set false only when the test image are well cropped' )
 
    parser.add_argument('--detector', default='fan', type=str, # fan or retinaface
                        help='detector for cropping face, check decalib/detectors.py for details' )
    # rendering option
    parser.add_argument('--rasterizer_type', default='pytorch3d', type=str,
                        help='rasterizer type: pytorch3d or standard' )
    parser.add_argument('--render_orig', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to render results in original image size, currently only works when rasterizer_type=standard')
    # save
    parser.add_argument('--useTex', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to use FLAME texture model to generate uv texture map, \
                            set it to True only if you downloaded texture model' )
    parser.add_argument('--extractTex', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to extract texture from input image as the uv texture map, set false if you want albeo map from FLAME mode' ) 
    main(parser.parse_args())