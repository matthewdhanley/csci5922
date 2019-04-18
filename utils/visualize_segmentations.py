import torch
from torchvision import transforms, datasets
import os
import argparse
import sys
from PIL import Image

sys.path.insert(0, '../')

from utils.data_loader import input_image_transform, output_image_transform
from utils.data_transforms import PILToLongTensor, LongTensorToRGBPIL
from models.UNet import UNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_segmentation(model, img_name):
    model.to(device)

    im = Image.open(img_name)
    image_tensor = input_image_transform(256)(im)
    image_tensor = image_tensor.unsqueeze_(0)
    image_tensor = image_tensor.to(device)

    segmentation_data = model(image_tensor)
    segmentation_data = segmentation_data.squeeze_(0)
    return segmentation_data


def save_segmentation(segmentation_tensor):
    pass


def main():
    parser = argparse.ArgumentParser(description='Visualize segmentations obtained from trained UNet')
    parser.add_argument('checkpoint',
                        type=str,
                        help='Path to UNet model checkpoint')
    parser.add_argument('img_path',
                        type=str,
                        help='Path to image or directory containing images')
    parser.add_argument('-o', '--out',
                        type=str,
                        help='Path to write segmentation visualizations to')
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        sys.exit('Specified checkpoint cannot be found')

    if not os.path.exists(args.img_path):
        sys.exit('Images for segmentation could not be found')

    imgs = []
    if os.path.isdir(args.img_path):
        for file in os.listdir(args.img_path):
            if os.path.isfile(os.path.join(args.img_path, file)):
                imgs.append(os.path.join(args.img_path, file))
    else:
        imgs.append(args.img_path)


    checkpoint = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)
    model = UNet(num_classes=len(datasets.Cityscapes.classes))
    model.load_state_dict(checkpoint['model_state_dict'])

    for img in imgs:
        segmentation_tensor = get_segmentation(model, img)
        save_segmentation(segmentation_tensor)

if __name__=='__main__':
    main()
    sys.exit(0)
