import torch.nn as nn
import torch
from torchvision import models
import torch.nn.functional as F
from . import resnet



class ExpressionLossNet(nn.Module):
    """ Code borrowed from EMOCA https://github.com/radekd91/emoca """
    def __init__(self):
        super(ExpressionLossNet, self).__init__()

        self.backbone = resnet.load_ResNet50Model() #out: 2048

        self.linear = nn.Sequential(
            nn.Linear(2048, 10))

    def forward2(self, inputs):
        inputs = F.interpolate(inputs, [224,224] , mode='bilinear')
        features = self.backbone(inputs)
        out = self.linear(features)
        return features, out

    def forward(self, inputs):
        # inputs = torch.reshape(inputs, (b,c, 224,224))
        inputs = F.interpolate(inputs, [224,224] , mode='bilinear')
        features = self.backbone(inputs)
        return features