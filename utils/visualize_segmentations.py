import torch
from torchvision import transforms, datasets
import os
import argparse
import sys
from PIL import Image
from data_transforms import PILToLongTensor, LongTensorToRGBPIL

sys.path.insert(0, '../models')
from UNet import UNet

def get_segmentation(model, img_name):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    #im = Image.open(img_name)
    #im.show()
    #image_tensor = tranforms.ToTensor()(im)
    #print(image_tensor)
    pass


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


    checkpoint = torch.load(checkpoint_path)
    model = UNet(num_classes=len(datasets.Cityscapes.classes))
    model.load_state_dict(checkpoint['model_state_dict'])

    for img in imgs:
        segmentation_tensor = get_segmentation(model, img)
        save_segmentation(segmentation_tensor)

if __name__=='__main__':
    main()
    sys.exit(0)
