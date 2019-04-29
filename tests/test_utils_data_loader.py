import unittest
import sys
import torch
from torchvision import datasets

sys.path.insert(0, '../')
from utils import data_loader

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.cityscapes_path = '../cityscapes'
        self.imagenet_path = '../tinyimagenet'

    def test_input_image_transform(self):
        pass

    def test_output_image_transform(self):
        pass

    def test_load_data(self):
        # Test loading of cityscapes dataset, no resize
        data = data_loader.load_data(self.cityscapes_path, resize=False)
        sample_img, sample_label = data[0]
        self.assertIsInstance(data, datasets.Cityscapes)
        self.assertEqual(sample_img.size(), torch.Size([3,1024,2048]))
        self.assertEqual(sample_label.size(), torch.Size([1024,2048]))

        # Test loading of cityscapes dataset, resizing data to 256x256
        data = data_loader.load_data(self.cityscapes_path, resize=True)
        sample_img, sample_label = data[0]
        self.assertIsInstance(data, datasets.Cityscapes)
        self.assertEqual(sample_img.size(), torch.Size([3,256,256]))
        self.assertEqual(sample_label.size(), torch.Size([256,256]))

        # Test loading of tinyimagenet dataset, no resize
        #data = data_loader.load_data(self.imagenet_path, resize=False)
        #sample_img, sample_label = data[0]
        #self.assertIsInstance(data, datasets.ImageFolder)
        #self.assertEqual(sample_img.size(), torch.Size([3,1024,2048]))
        #self.assertEqual(sample_label.size(), torch.Size([1024,2048]))

        # Test loading of tinyimagenet dataset, resizing data to 256x256
        #data = data_loader.load_data(self.imagenet_path, resize=True)
        #sample_img, sample_label = data[0]
        #self.assertIsInstance(data, datasets.ImageFolder)
        #self.assertEqual(sample_img.size(), torch.Size([3,256,256]))
        #self.assertEqual(sample_label.size(), torch.Size([256,256]))



if __name__ == '__main__':
    unittest.main()
