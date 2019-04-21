'''
Filter visualization techniques derived from those discussed in
'How to Visualize Convolutional Features in 40 Lines of Code', by Fabio M. Graetz.

https://towardsdatascience.com/how-to-visualize-convolutional-features-in-40-lines-of-code-70b7d87b0030
'''
import torch
from torchvision import datasets, models
from torch.autograd import Variable
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
import sys
import os

from models.UNet import UNet

def upscale_image(image, upscale_size):
    # Upscales an image with dimensions (3,h,w) to (3,upscale_size,upscale_size)
    image = np.transpose(image, axes=(1,2,0))
    image = cv2.resize(image, (upscale_size,upscale_size), interpolation = cv2.INTER_CUBIC)
    image = np.transpose(image, axes=(2,0,1))
    return image


class LayerActivations():
    def __init__(self, layer):
        self.forward_hook = layer.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.activations = output

    def remove_hook(self):
        self.forward_hook.remove()


class LayerActivationAnalysis():
    def __init__(self, model, layer):
        self.model = model
        self.layer = layer

    def set_layer(self, layer):
        self.layer = layer

    def get_activated_filter_indices(self, initial_img_size=56):
        layer_activations = LayerActivations(self.layer)
        image = (np.random.uniform(0, 255, size=(3,initial_img_size,initial_img_size)) / 255).astype(np.float32, copy=False)
        image_tensor = torch.from_numpy(image).expand(1, -1, -1, -1)

        _ = self.model(image_tensor)

        filter_activations = layer_activations.activations[0].detach().numpy()

        layer_activations.remove_hook()

        return np.unique(np.nonzero(filter_activations)[0])

    def get_max_activating_image(self, filter_index, initial_img_size=56, upscaling_steps=12, upscaling_factor=1.2, lr=0.01, update_steps=15, verbose=False):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        layer_activations = LayerActivations(self.layer)
        size = initial_img_size
        image = (np.random.uniform(0, 255, size=(3,size,size)) / 255).astype(np.float32, copy=False)

        for i in range(upscaling_steps):
            image_tensor = torch.from_numpy(image).expand(1, -1, -1, -1)
            image_tensor = Variable(image_tensor, requires_grad=True)
            image_tensor = image_tensor.to(device)

            if not image_tensor.grad is None:
                image_tensor.grad.zero_()

            optimizer = torch.optim.Adam([image_tensor], lr=lr, weight_decay=1e-6)

            # Update image update_steps times
            for n in range(update_steps):
                optimizer.zero_grad()
                _ = self.model(image_tensor)
                loss = -1 * (layer_activations.activations[0, filter_index].mean())
                if verbose and (n % 5 == 0):
                    print('Loss at upscale step {}/{}, update {}/{}: {}'
                            .format(i, upscaling_steps-1, n, update_steps-1, loss))
                loss.backward()
                optimizer.step()

            image = torch.squeeze(image_tensor, dim=0).clone().detach().numpy()
            size = int(upscaling_factor * size)
            image = upscale_image(image, size)

        output = np.transpose(image, axes=(1,2,0))
        layer_activations.remove_hook()
        return np.clip(output, 0, 1)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Analyze encoder module filters of a trained model')

    parser.add_argument('--unet',
                        action='store_true',
                        help='Analyze UNet encoder')

    parser.add_argument('--ckpt',
                        required='--unet' in sys.argv,
                        type=str,
                        help='Path to UNet model checkpoint')

    parser.add_argument('--vgg11',
                        action='store_true',
                        help='Analyze VGG11 encoder (pre-trained on ImageNet)')

    parser.add_argument('-o', '--out',
                        type=str,
                        default='./',
                        help='Path to write segmentation visualizations to')
    args = parser.parse_args()

    if args.unet:
        if not os.path.exists(args.ckpt):
            sys.exit('Specified checkpoint cannot be found')

        checkpoint = torch.load('models/unet_100.tar', map_location=lambda storage, loc: storage)
        model = UNet(num_classes=len(datasets.Cityscapes.classes), encoder_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        layer = model.encoder6
    elif args.vgg11:
        model = models.vgg11(pretrained=True)
        layer = list(model.children())[0][11]
    else:
        sys.exit('No model provided, please specify --unet or --vgg11 to analyze the UNet or VGG11 encoder, respectively')

    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    analyzer = LayerActivationAnalysis(model, layer)

    img = analyzer.get_max_activating_image(6, verbose=True)
    plt.figure(figsize=(7,7))
    plt.imshow(img)
    plt.show()
