'''
UNet implementation based off of the TernausNet Architecture.

TernausNet: U-Net with VGG11 Encoder Pre-Trained on ImageNet for Image Segmentation
https://arxiv.org/abs/1801.05746
'''

import torch
import torch.nn as nn
from torchvision import models


class UNetDecoderModule(nn.Module):
    def __init__(self, in_channels, in_channels_ct, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.in_channels_ct = in_channels_ct
        self.out_channels = out_channels

        self.conv = nn.Conv2d(in_channels, in_channels_ct, kernel_size=3, padding=1)
        self.transpose_conv = nn.ConvTranspose2d(in_channels_ct, out_channels, kernel_size=3,
                                                 stride=2, padding=1, output_padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, X):
        output = self.conv(X)
        output = self.act(output)
        output = self.transpose_conv(output)
        output = self.act(output)
        return output


class UNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.name = "UNet"

        self.num_classes = num_classes

        vgg11_encoder = models.vgg11(pretrained=False).features
        self.encoder1 = vgg11_encoder[0]
        self.encoder2 = vgg11_encoder[3]
        self.encoder3 = vgg11_encoder[6]
        self.encoder4 = vgg11_encoder[8]
        self.encoder5 = vgg11_encoder[11]
        self.encoder6 = vgg11_encoder[13]
        self.encoder7 = vgg11_encoder[16]
        self.encoder8 = vgg11_encoder[18]
        self.encoder_act = vgg11_encoder[1]
        self.max_pool = vgg11_encoder[2]

        self.decoder6 = UNetDecoderModule(512, 512, 256)
        self.decoder5 = UNetDecoderModule(256 + self.encoder8.out_channels, 512, 256)
        self.decoder4 = UNetDecoderModule(256 + self.encoder6.out_channels, 512, 128)
        self.decoder3 = UNetDecoderModule(128 + self.encoder4.out_channels, 256, 64)
        self.decoder2 = UNetDecoderModule(64 + self.encoder2.out_channels, 128, 32)
        self.decoder1 = nn.Sequential(
            nn.Conv2d(32 + self.encoder1.out_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, self.num_classes, kernel_size=1),
        )

    def forward(self, X):
        '''
        Encode
        '''
        # Skip connection
        encoder1_out = self.encoder_act(self.encoder1(X))
        encoder1_out_pooled = self.max_pool(encoder1_out)

        # Skip connection
        encoder2_out = self.encoder_act(self.encoder2(encoder1_out_pooled))
        encoder2_out_pooled = self.max_pool(encoder2_out)

        encoder3_out = self.encoder_act(self.encoder3(encoder2_out_pooled))
        # Skip connection
        encoder4_out = self.encoder_act(self.encoder4(encoder3_out))
        encoder4_out_pooled = self.max_pool(encoder4_out)

        encoder5_out = self.encoder_act(self.encoder5(encoder4_out_pooled))
        # Skip connection
        encoder6_out = self.encoder_act(self.encoder6(encoder5_out))
        encoder6_out_pooled = self.max_pool(encoder6_out)

        encoder7_out = self.encoder_act(self.encoder7(encoder6_out_pooled))
        # Skip connection
        encoder8_out = self.encoder_act(self.encoder8(encoder7_out))
        encoder8_out_pooled = self.max_pool(encoder8_out)

        '''
        Decode
        '''
        decoder6_out = self.decoder6(encoder8_out_pooled)
        decoder5_out = self.decoder5(torch.cat([decoder6_out, encoder8_out], 1))
        decoder4_out = self.decoder4(torch.cat([decoder5_out, encoder6_out], 1))
        decoder3_out = self.decoder3(torch.cat([decoder4_out, encoder4_out], 1))
        decoder2_out = self.decoder2(torch.cat([decoder3_out, encoder2_out], 1))
        decoder1_out = self.decoder1(torch.cat([decoder2_out, encoder1_out], 1))
        self.activs = [X, encoder1_out, encoder1_out_pooled,
                       encoder2_out, encoder2_out_pooled,
                       encoder3_out,
                       encoder4_out, encoder4_out_pooled,
                       encoder5_out,
                       encoder6_out, encoder6_out_pooled,
                       encoder7_out,
                       encoder8_out, encoder8_out_pooled,
                       decoder6_out,
                       decoder5_out,
                       decoder4_out,
                       decoder3_out,
                       decoder2_out,
                       decoder1_out]

        return decoder1_out
