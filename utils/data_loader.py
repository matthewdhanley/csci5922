from torchvision import transforms, datasets
from utils.data_transforms import PILToLongTensor
from PIL import Image


def input_image_transform(resize_size):
    if resize_size != 0:
        transform = transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.ToTensor()
        ])
    else:
        transforms.Compose([
            transforms.ToTensor()
        ])

    return transform


def output_image_transform(resize_size):
    if resize_size != 0:
        transform = transforms.Compose([
            transforms.Resize((resize_size, resize_size), Image.NEAREST),
            PILToLongTensor()
        ])
    else:
        transform = transforms.Compose([
            PILToLongTensor()
        ])

    return transform


def load_data(path, resize=True):
    """
    Loads Cityscapes data from path input via command line.
    :param path: Path to root of Cityscapes directory
    :param resize: Set to true to size down the images. Default=True
    :return: torchvision cityscapes dataset object
    """

    if resize:
        resize_size = 256
        print("Resizing images to {}x{}".format(resize_size, resize_size))

        input_transform = input_image_transform(resize_size)
        output_transform = output_image_transform(resize_size)

    else:
        input_transform = input_image_transform(0)
        output_transform = output_image_transform(0)

    dataset = datasets.Cityscapes(path,
                                  split='train',
                                  mode='fine',
                                  target_type='semantic',
                                  transform=input_transform,
                                  target_transform=output_transform)

    return dataset
