import argparse


def get_cli_arguments():
    """
    Defines command-line arguments, and parses them.
    """
    parser = argparse.ArgumentParser(description='Train UNet on CityScapes data')
    parser.add_argument('path',
                        type=str,
                        help='Relative path to directory containing to CityScapes gtFine and leftImg8bit directories')

    parser.add_argument("--mode", "-m",
                        choices=['train', 'test'],
                        default='train',
                        help="train: performs training. Test tests the model. Default: train")

    parser.add_argument('--checkpoint',
                        type=str,
                        default=None,
                        help='Relative path to saved checkpoint')

    parser.add_argument('--subset',
                        action='store_true',
                        help='Run with a small subset of the data. This will result in faster execution')

    parser.add_argument('--no_resize',
                        action='store_true',
                        default=False,
                        help='Do not resize input images. Results in significantly more memory needing to be allocated.'
                             'Default=False')

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