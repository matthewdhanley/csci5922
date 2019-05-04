import unittest
import sys
import os
import numpy as np
import torch
from torchvision import transforms, datasets
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import data_loader


class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.cityscapes_path = os.path.join(os.path.dirname(__file__), '../data')
        self.imagenet_path = os.path.join(os.path.dirname(__file__), 'tinyimagenet_test')
        self.image = Image.fromarray(np.uint8(np.random.randint(0, 256, (32, 32))))

    def test_input_image_transform(self):
        # Test with no resizing
        transform = data_loader.input_image_transform(0)
        image_tensor = transform(self.image)
        self.assertIsInstance(transform, transforms.Compose)
        self.assertIsInstance(image_tensor, torch.Tensor)
        self.assertEqual(image_tensor.size(), torch.Size([1, 32, 32]))

        # Test with resizing to 16x16
        transform = data_loader.input_image_transform(16)
        image_tensor = transform(self.image)
        self.assertIsInstance(transform, transforms.Compose)
        self.assertIsInstance(image_tensor, torch.Tensor)
        self.assertEqual(image_tensor.size(), torch.Size([1, 16, 16]))

    def test_output_image_transform(self):
        # Test with no resizing
        transform = data_loader.output_image_transform(0)
        image_tensor = transform(self.image)
        self.assertIsInstance(transform, transforms.Compose)
        self.assertIsInstance(image_tensor, torch.Tensor)
        self.assertEqual(image_tensor.size(), torch.Size([32, 32]))

        # Test with resizing to 16x16
        transform = data_loader.output_image_transform(16)
        image_tensor = transform(self.image)
        self.assertIsInstance(transform, transforms.Compose)
        self.assertIsInstance(image_tensor, torch.Tensor)
        self.assertEqual(image_tensor.size(), torch.Size([16, 16]))

    def test_load_data(self):
        # Test loading of cityscapes dataset, no resize
        data = data_loader.load_data(self.cityscapes_path, 'cityscapes', resize=False)
        sample_img, sample_label = data[0]
        self.assertIsInstance(data, datasets.Cityscapes)
        self.assertEqual(sample_img.size(), torch.Size([3, 1024, 2048]))
        self.assertEqual(sample_label.size(), torch.Size([1024, 2048]))

        # Test loading of cityscapes dataset, resizing data to 256x256
        data = data_loader.load_data(self.cityscapes_path, 'cityscapes', resize=True)
        sample_img, sample_label = data[0]
        self.assertIsInstance(data, datasets.Cityscapes)
        self.assertEqual(sample_img.size(), torch.Size([3, 256, 256]))
        self.assertEqual(sample_label.size(), torch.Size([256, 256]))

        # Test loading of tinyimagenet dataset, no resize
        data = data_loader.load_data(self.imagenet_path, 'imagenet', resize=False)
        sample_img, sample_label = data[0]
        self.assertIsInstance(data, datasets.ImageFolder)
        self.assertEqual(sample_img.size(), torch.Size([3, 64, 64]))
        self.assertIsInstance(sample_label, int)

        # Test loading of tinyimagenet dataset, resizing data to 256x256
        data = data_loader.load_data(self.imagenet_path, 'imagenet', resize=True)
        sample_img, sample_label = data[0]
        self.assertIsInstance(data, datasets.ImageFolder)
        self.assertEqual(sample_img.size(), torch.Size([3, 256, 256]))
        self.assertIsInstance(sample_label, int)


if __name__ == '__main__':
    unittest.main()
