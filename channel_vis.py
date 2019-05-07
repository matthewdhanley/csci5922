import torch
from torchvision import datasets, models
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
import sys
import os

from utils.activation_analysis import LayerActivationAnalysis
from models.UNet import UNet

# Maps an encoder layer index to the index of the child module associated with
# that layer in the VGG11 model
vgg11_layer_dict = {1:0, 2:3, 3:6, 4:8, 5:11, 6:13, 7:16, 8:18}

def get_conv_layer(model, layer_index):
    '''
    Gets the conv layer torch module associated with a given layer index
    '''
    if isinstance(model, UNet):
        layer = model.get_encoder_layer(layer_index)
    elif isinstance(model, models.vgg.VGG):
        child_index = vgg11_layer_dict[layer_index]
        layer = list(model.children())[0][child_index]
    return layer

def save_image(image_array, destination):
    plt.figure()
    plt.imshow(image_array)
    plt.axis('off')
    plt.savefig(destination, transparent=True, bbox_inches='tight', pad_inches=0)
    plt.close()

def save_image_grid(image_arrays, destination, width=16, height=16):
    fig = plt.figure(figsize=(width, height))
    rows = np.ceil(np.sqrt(len(image_arrays)))
    cols = np.ceil(np.sqrt(len(image_arrays)))
    for i, img in enumerate(image_arrays):
        fig.add_subplot(rows, cols, i+1)
        plt.imshow(img)
        plt.axis('off')
        plt.subplots_adjust(wspace=0, hspace=0.06)
    plt.savefig(destination, transparent=True, bbox_inches='tight', pad_inches=0)
    plt.close()


def channel_vis_driver(model_name, checkpoint_path, data_path, dataset, conv_layer, channels, init_img_size, upscale_steps, upscale_factor, lr, update_steps, grid, out_path, verbose):
    if model_name == 'unet':
        if not os.path.exists(checkpoint_path):
            sys.exit('Specified checkpoint cannot be found')

        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        model = UNet(num_classes=len(datasets.Cityscapes.classes), encoder_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
    elif model_name == 'vggmod':
        model = models.vgg11(pretrained=True)
    else:
        sys.exit('No model provided, please specify --unet or --vgg11 to analyze the UNet or VGG11 encoder, respectively')

    # Set model to evaluation mode and fix the parameter values
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    layer = get_conv_layer(model, conv_layer)

    analyzer = LayerActivationAnalysis(model, layer)


    # Save a grid of channel activation visualizations
    if grid:
        if not channels:
            # Get a random sample of 9 activated channels
            channels = analyzer.get_activated_filter_indices()
            np.random.shuffle(channels)
            channels = channels[:9]

        imgs = []
        for i, channel in enumerate(channels):
            if verbose:
                print('Generating image {} of {}...'.format(i+1, len(channels)))

            img = analyzer.get_max_activating_image(channel,
                                                    initial_img_size=init_img_size,
                                                    upscaling_steps=upscale_steps,
                                                    upscaling_factor=upscale_factor,
                                                    lr=lr,
                                                    update_steps=update_steps,
                                                    verbose=verbose)

            imgs.append(img)
        channel_string = '-'.join(str(channel_id) for channel_id in channels)
        output_dest = os.path.join(out_path, '{}_layer{}_channels{}.png'.format(model_name, conv_layer, channel_string))
        save_image_grid(imgs, output_dest)

    # Save a channel activation visualization for each specified channel
    elif channels is not None:
        for channel in channels:

            img = analyzer.get_max_activating_image(channel,
                                                    initial_img_size=init_img_size,
                                                    upscaling_steps=upscale_steps,
                                                    upscaling_factor=upscale_factor,
                                                    lr=lr,
                                                    update_steps=update_steps,
                                                    verbose=verbose)

            output_dest = os.path.join(out_path, '{}_layer{}_channel{}.png'.format(model_name, conv_layer, channel))
            save_image(img, output_dest)

    else:
        # Compute the average number number of channels activated in each layer
        if data_path and dataset:
            layers = [get_conv_layer(model, i) for i in [1,2,3,4,5,6,7,8]]
            avg = analyzer.get_avg_activated_channels(layers, data_path, dataset, 100)
            print('Average number of channels activated per convolutional layer: {}'.format(avg))

        # Output the channels activated by a randomly initialize image
        else:
            activated_channels = analyzer.get_activated_filter_indices(initial_img_size=init_img_size)
            print('Output channels in conv layer {} activated by random image input:'.format(conv_layer))
            print(activated_channels)
            print()
            print('(Total of {} activated channels)'.format(len(activated_channels)))
