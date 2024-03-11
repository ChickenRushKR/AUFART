import numpy as np
from skimage.io import imread
from decalib.models.gat import GAT
import cv2
import torch
imagepath = '/mnt/hdd/dataset/FFHQ/images/00000.jpg'
image = np.array(imread(imagepath))
image = cv2.resize(image, (224,224))
image = image / 255.
image = torch.tensor(image).unsqueeze(0).permute(0,3,1,2).to('cuda')

gatmodule = GAT(nfeat=768, 
        nhid=768, 
        noutput=159, 
        dropout=0.0, 
        nheads=8, 
        alpha=0.2,
        batchsize=1)

_ = gatmodule(image)
print(_)

