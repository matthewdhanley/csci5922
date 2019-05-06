import argparse


def get_cli_arguments():
    """
    Defines command-line arguments, and parses them.
    """
    parser = argparse.ArgumentParser(description='Train UNet on CityScapes data')
    parser.add_argument('path',
                        type=str,
                        help='Relative path to data directory containing either the CityScapes'
                             'gtFine and leftImg8bit directories, tinyimagenet train directory,'
                             'or activation pickle file.')

    parser.add_argument('--dataset',
                        type=str,
                        choices=['cityscapes', 'imagenet'],
                        default='cityscapes',
                        help='Specify which dataset is located at path argument. Default: cityscapes')

    parser.add_argument("--mode", "-m",
                        choices=['train', 'test', 'activations', 'compare_activations', 'view_activations'],
                        default='train',
                        help="train: performs training. Test tests the model. activations saves the activations. Must "
                             "use argument --model with activations keyword. compare_activations comapres the "
                             "activations in the specified folder. view_activations views the activations in the"
                             "specified folder. Default: train")

    parser.add_argument("--model",
                        choices=['unet', 'vggmod', 'both'],
                        default=None,
                        help="Specifiy which model to use to save activations. unet uses the custom UNet, vggmod uses"
                             "the pre-trained VGG11 model. Use --checkpoint to specify a pretrained model for UNet.")

    parser.add_argument('--checkpoint',
                        type=str,
                        default=None,
                        help='Relative path to saved checkpoint')

    parser.add_argument('--batch_num',
                        type=int,
                        default=0,
                        help='Specify which batch to visualize. Used with \"--mode view_activations\"')

    parser.add_argument('--start_layer',
                        type=int,
                        default=0,
                        help='Specify which layer to begin visualizations. Used with \"--mode view_activations\"')

    parser.add_argument('--stop_layer',
                        type=int,
                        default=22,
                        help='Specify which layer to begin visualizations. Used with \"--mode view_activations\"')

    parser.add_argument('--subset',
                        action='store_true',
                        help='Run with a small subset of the data. This will result in faster execution')

    parser.add_argument('--no_resize',
                        action='store_true',
                        default=False,
                        help='Do not resize input images. Results in significantly more memory needing to be allocated.'
                             'Default=False')

    parser.add_argument('--pretrained',
                        action='store_true',
                        default=False,
                        help='Train the UNet with a pretrained VGG11 encoder. Weights of the encoder will not be'
                             'updated.')

    parser.add_argument('--savedir',
                        type=str,
                        default=None,
                        help='Relative path to location to save checkpoint')

    parser.add_argument('--type',
                        choices=['normal', 'blur', 'dilation', 'pooling'],
                        default='normal',
                        help='Type of comparison to use for channel matching.')

    # Hyperparameters.
    parser.add_argument('-b', '--batch_size',
                        type=int,
                        default=8,
                        help='Image batch size (default 0.001)')

    parser.add_argument('-lr', '--learning_rate',
                        type=float,
                        default=0.001,
                        help='Learning rate (default 0.001)')

    parser.add_argument('--epochs',
                        type=int,
                        default=100,
                        help='Maximum number of training epochs')

    return parser.parse_args()
