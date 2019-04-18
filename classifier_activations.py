# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 14:18:13 2019

@author: Scott Lucas Young
"""
import sys
import torch
import torch.nn as nn
from torchvision import transforms, datasets, models
import numpy as np
from PIL import Image


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

class PILToLongTensor(object):
    """Converts a ``PIL Image`` to a ``torch.LongTensor``.
    Code adapted from: http://pytorch.org/docs/master/torchvision/transforms.html?highlight=totensor
    """

    def __call__(self, pic):
        """Performs the conversion from a ``PIL Image`` to a ``torch.LongTensor``.
        Keyword arguments:
        - pic (``PIL.Image``): the image to convert to ``torch.LongTensor``
        Returns:
        A ``torch.LongTensor``.
        """
        if not isinstance(pic, Image.Image):
            raise TypeError("pic should be PIL Image. Got {}".format(
                type(pic)))

        # handle numpy array
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            # backward compatibility
            return img.long()

        # Convert PIL image to ByteTensor
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))

        # Reshape tensor
        nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        # Convert to long and squeeze the channels
        return img.transpose(0, 1).transpose(0,2).contiguous().long().squeeze_()



def load_data(path):
    input_transform = transforms.Compose([
        # transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    output_transform = transforms.Compose([
        # transforms.Resize((256, 256), Image.NEAREST),
        PILToLongTensor()
    ])

    dataset = datasets.Cityscapes(path, split='train', mode='fine',
                                  target_type='semantic', transform=input_transform,
                                  target_transform=output_transform)

    return dataset




def main():
    net = VGGmod()
    dataset = load_data("Cityscapes/")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    for inp, _ in dataloader:
        output = net(inp)
        print(output)
        print(net.activs)
    
    
if __name__ == '__main__':
    main()
    sys.exit(0)