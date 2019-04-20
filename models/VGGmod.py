# -*- coding: utf-8 -*-
import torch.nn as nn
from torchvision import models


class VGGmod(nn.Module):

    def __init__(self, num_classes=1000, init_weights=True):
        super().__init__()

        self.num_classes = num_classes

        self.features = models.vgg11(pretrained=True).features

    def forward(self, x):
        self.activs = []
        for layer in self.features:
            self.activs.append(x)
            x = layer(x)
        self.activs.append(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
