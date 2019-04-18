import torch
from torchvision import transforms, datasets
import os
import argparse
import sys
from PIL import Image
from collections import OrderedDict

sys.path.insert(0, '../')

from utils.data_loader import input_image_transform, output_image_transform
from utils.data_transforms import PILToLongTensor, LongTensorToRGBPIL
from models.UNet import UNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CityscapeSegmentationVis():
    def __init__(self, model, input_transform):
        self.model = model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.input_transform = input_transform

        # LongTensorToRGBPIL is instantiated with an ordered dictionary of tuples that
        # map class names to RGB colors.  It assumes that the index of the item in the
        # dictionary corresponds with the id of the class.  However, the CityScapes
        # class 'licencse plate' has an id of -1.  For this reason, the shift in
        # the ordered dictionary below is required.
        ordered_cscape_classes = sorted(datasets.Cityscapes.classes, key=lambda tup: tup.id)
        cscape_class_tuples = [(c.name, c.color) for c in ordered_cscape_classes][1:]
        cscape_class_tuples.append((ordered_cscape_classes[0].name, ordered_cscape_classes[0].color))
        self.segmentation_transform = LongTensorToRGBPIL(OrderedDict(cscape_class_tuples))

    def get_predicted_segmentation(self, img_name):
        self.model.to(self.device)

        im = Image.open(img_name)
        image_tensor = self.input_transform(im)
        image_tensor = image_tensor.unsqueeze_(0)
        image_tensor = image_tensor.to(self.device)

        segmentation_data = self.model(image_tensor)
        predicted_classes = torch.argmax(segmentation_data, dim=1)
        predicted_classes = predicted_classes.squeeze_(0)

        result = self.segmentation_transform(predicted_classes)
        result.show()
        return result

    def save_segmentation(self, segmentation_tensor):
        pass


def main():
    parser = argparse.ArgumentParser(description='Visualize segmentations obtained from trained UNet')
    parser.add_argument('checkpoint',
                        type=str,
                        help='Path to UNet model checkpoint')
    parser.add_argument('img_path',
                        type=str,
                        help='Path to image or directory containing images to segment')
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
    model.eval()

    visualizer = CityscapeSegmentationVis(model, input_image_transform(256))

    for img in imgs:
        segmentation_tensor = visualizer.get_predicted_segmentation(img)
        visualizer.save_segmentation(segmentation_tensor)

if __name__=='__main__':
    main()
    sys.exit(0)
