import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import math
from .swin_transformer import swin_transformer_tiny, swin_transformer_small, swin_transformer_base
from .resnet import resnet18, resnet50, resnet101
from .graph import create_e_matrix
from .graph_edge_model import GEM
from .basic_block import *
from .modeling_pretrain import pretrain_mae_base_patch16_224

class Head(nn.Module):
    def __init__(self, in_channels, num_main_classes = 27, num_sub_classes = 14):
        super(Head, self).__init__()
        self.in_channels = in_channels
        self.num_main_classes = num_main_classes
        self.num_sub_classes = num_sub_classes

        main_class_linear_layers = []

        for i in range(self.num_main_classes):
            layer = LinearBlock(self.in_channels, self.in_channels)
            main_class_linear_layers += [layer]
        self.main_class_linears = nn.ModuleList(main_class_linear_layers)

        self.edge_extractor = GEM(self.in_channels, num_main_classes)

        self.relu = nn.ReLU()


    def forward(self, x):
        # AFG
        f_u = []
        for i, layer in enumerate(self.main_class_linears):
            f_u.append(layer(x).unsqueeze(1))
        f_u = torch.cat(f_u, dim=1)
        f_v = f_u.mean(dim=-2)

        f_e = self.edge_extractor(f_u, x)
        f_e = f_e.mean(dim=-2)
        # f_v, f_e = self.gnn(f_v, f_e)
        return f_v, f_e



class MEFARG(nn.Module):
    def __init__(self, num_main_classes = 27, num_sub_classes = 14, backbone='swin_transformer_base'):
        super(MEFARG, self).__init__()
        if 'transformer' in backbone:
            if backbone == 'swin_transformer_tiny':
                self.backbone = swin_transformer_tiny()
            elif backbone == 'swin_transformer_small':
                self.backbone = swin_transformer_small()
            else:
                self.backbone = swin_transformer_base()
            self.in_channels = self.backbone.num_features
            self.out_channels = self.in_channels // 2
            self.backbone.head = None
        elif 'resnet' in backbone:
            if backbone == 'resnet18':
                self.backbone = resnet18()
            elif backbone == 'resnet101':
                self.backbone = resnet101()
            else:
                self.backbone = resnet50()
            self.in_channels = self.backbone.fc.weight.shape[1]
            # self.out_channels = self.in_channels // 4
            self.out_channels = self.in_channels // 4
            self.backbone.fc = None
        elif 'mae' in backbone:
            self.backbone = pretrain_mae_base_patch16_224()
            self.in_channels = self.backbone.embed_dim
            self.out_channels = self.in_channels // 2
        else:
            raise Exception("Error: wrong backbone name: ", backbone)

        self.global_linear = LinearBlock(self.in_channels, self.out_channels)
        self.head = Head(self.out_channels, num_main_classes, num_sub_classes)
        self.global_linear.eval()
        self.head.eval()

    def forward(self, x):
        # x: b d c
        x = self.backbone(x)
        x = self.global_linear(x)
        f_v, f_e = self.head(x)
        return f_v, f_e