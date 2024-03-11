import cv2
import os
from glob import glob
# inputpath = '/media/cine/First/TransformerCode/TestNewIdea/TestReult/pretrainNewIdea_236B/35-30-1920x1080_sequence/18X/result'
name = 'Actor_03sad_texture' # # angry, calm, disgust, fear, happy, neut, sad, surprise
inputpath = '/mnt/hdd/EncoderTrainingCode/Code/videoResult/testGATE20_now/result/'
imagepaths = sorted(glob(os.path.join(inputpath, '*.*')))
savefolder = '/mnt/hdd/EncoderTrainingCode/Code/videoResult/testGATE20_now'
os.makedirs(savefolder, exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter(os.path.join(savefolder, name+".mp4"), fourcc, 1, (448 * 3, 448), True)

for i, imagepath in enumerate(imagepaths):
    for _ in range(4):
        vis_image1 = cv2.imread(imagepath)
        out.write(vis_image1)
out.release()
