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

def get_cli_arguments():
    parser = argparse.ArgumentParser(description='Analyze encoder module filters of a trained model')

    parser.add_argument('--unet',
                        action='store_true',
                        help='Analyze UNet encoder')

    parser.add_argument('--checkpoint',
                        required='--unet' in sys.argv,
                        type=str,
                        help='Path to UNet model checkpoint')

    parser.add_argument('--vgg11',
                        action='store_true',
                        help='Analyze VGG11 encoder (pre-trained on ImageNet)')

    parser.add_argument('--layer',
                        type=int,
                        default=8,
                        choices=[1,2,3,4,5,6,7,8],
                        help='Layer of encoder to analyze')

    parser.add_argument('--channels',
                        type=int,
                        nargs='*',
                        help='Layer output channels to visualize')

    parser.add_argument('--img_size',
                        type=int,
                        default=56,
                        help='Initial size of the randomly initialized image')

    parser.add_argument('--upscale_steps',
                        type=int,
                        default=12,
                        help='Number of upscaling steps while optimizing the maximally activating image')

    parser.add_argument('--upscale_factor',
                        type=float,
                        default=1.2,
                        help='Upscaling factor at each upscaling step')

    parser.add_argument('--lr',
                        type=float,
                        default=0.01,
                        help='Learning rate for image optimization update step')

    parser.add_argument('--steps',
                        type=int,
                        default=15,
                        help='Number of optimization update steps at each upscaling step')

    parser.add_argument('--grid',
                        action='store_true',
                        help='Visualize multiple output channels as a grid')

    parser.add_argument('-o', '--out',
                        type=str,
                        default='./',
                        help='Path to write visualizations to')

    parser.add_argument('--data_path',
                        type=str,
                        help='Relative path to data directory containing either the CityScapes'
                             'gtFine and leftImg8bit directories or tinyimagenet train directory')

    parser.add_argument('--dataset',
                        type=str,
                        choices=['cityscapes', 'imagenet'],
                        default='cityscapes',
                        help='Specify which dataset is located at path argument. Default: cityscapes')

    parser.add_argument('--sample_size',
                        type=int,
                        default=50,
                        help='Number of input samples to use from the specified dataset'
                             'when computing average number of activated channels')

    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        help='Output status of image optimization processes')
    return parser.parse_args()


if __name__=='__main__':
    args = get_cli_arguments()

    if args.unet:
        if not os.path.exists(args.checkpoint):
            sys.exit('Specified checkpoint cannot be found')

        checkpoint = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)
        model = UNet(num_classes=len(datasets.Cityscapes.classes), encoder_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        model_name = 'unet'
    elif args.vgg11:
        model = models.vgg11(pretrained=True)
        model_name = 'vgg11'
    else:
        sys.exit('No model provided, please specify --unet or --vgg11 to analyze the UNet or VGG11 encoder, respectively')

    # Set model to evaluation mode and fix the parameter values
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    layer = get_conv_layer(model, args.layer)

    analyzer = LayerActivationAnalysis(model, layer)


    # Save a grid of channel activation visualizations
    if args.grid:
        if args.channels is not None:
            channels = args.channels
        else:
            # Get a random sample of 9 activated channels
            channels = analyzer.get_activated_filter_indices()
            np.random.shuffle(channels)
            channels = channels[:9]

        imgs = []
        for i, channel in enumerate(channels):
            if args.verbose:
                print('Generating image {} of {}...'.format(i+1, len(channels)))

            img = analyzer.get_max_activating_image(channel,
                                                    initial_img_size=args.img_size,
                                                    upscaling_steps=args.upscale_steps,
                                                    upscaling_factor=args.upscale_factor,
                                                    lr=args.lr,
                                                    update_steps=args.steps,
                                                    verbose=args.verbose)

            imgs.append(img)
        channel_string = '-'.join(str(channel_id) for channel_id in channels)
        output_dest = os.path.join(args.out, '{}_layer{}_channels{}.png'.format(model_name, args.layer, channel_string))
        save_image_grid(imgs, output_dest)

    # Save a channel activation visualization for each specified channel
    elif args.channels is not None:
        for channel in args.channels:

            img = analyzer.get_max_activating_image(channel,
                                                    initial_img_size=args.img_size,
                                                    upscaling_steps=args.upscale_steps,
                                                    upscaling_factor=args.upscale_factor,
                                                    lr=args.lr,
                                                    update_steps=args.steps,
                                                    verbose=args.verbose)

            output_dest = os.path.join(args.out, '{}_layer{}_channel{}.png'.format(model_name, args.layer, channel))
            save_image(img, output_dest)

    else:
        # Compute the average number number of channels activated in each layer
        if args.data_path and args.dataset:
            layers = [get_conv_layer(model, i) for i in [1,2,3,4,5,6,7,8]]
            avg = analyzer.get_avg_activated_channels(layers, args.data_path, args.dataset, args.sample_size)
            print('Average number of channels activated per convolutional layer: {}'.format(avg))

        # Output the channels activated by a randomly initialize image
        else:
            activated_channels = analyzer.get_activated_filter_indices(initial_img_size=args.img_size)
            print('Output channels in conv layer {} activated by random image input:'.format(args.layer))
            print(activated_channels)
            print()
            print('(Total of {} activated channels)'.format(len(activated_channels)))
