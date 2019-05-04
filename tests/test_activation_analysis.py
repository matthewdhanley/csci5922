import unittest
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.activation_analysis import LayerActivationAnalysis, upscale_image
from models.VGGmod import VGGmod


class TestActivationAnalysis(unittest.TestCase):
    def setUp(self):
        self.model = VGGmod()
        self.layer0 = list(self.model.children())[0][7]
        self.layer1 = list(self.model.children())[0][5]
        self.layer_analyzer = LayerActivationAnalysis(self.model, layer=self.layer0)

    def test_set_layer(self):
        # Tests LayerActivationAnalysis.set_layer(...)
        layer_analyzer = LayerActivationAnalysis(self.model, layer=self.layer0)
        layer_analyzer.set_layer(self.layer1)
        self.assertEqual(layer_analyzer.layer, self.layer1)

    def test_get_activated_filter_indices(self):
        # Tests LayerActivationAnalysis.get_activated_filter_indices(...)
        activated_channels = self.layer_analyzer.get_activated_filter_indices()
        self.assertIsInstance(activated_channels, np.ndarray)

    def test_get_max_activating_image(self):
        # Tests LayerActivationAnalysis.get_max_activating_image(...)
        img = self.layer_analyzer.get_max_activating_image(channel_index=1, initial_img_size=64, upscaling_steps=1,
                                                           upscaling_factor=1.2)
        self.assertIsInstance(img, np.ndarray)
        self.assertEqual(img.shape, (int(64 * 1.2), int(64 * 1.2), 3))

    def test_upscale_image(self):
        # Tests activation_analysis.upscale_image(...)
        img = (np.random.uniform(0, 255, size=(3, 32, 32)) / 255).astype(np.float32, copy=False)
        upscaled_image = upscale_image(img, 64)
        self.assertIsInstance(img, np.ndarray)
        self.assertEqual(upscaled_image.shape, (3, 64, 64))


if __name__ == '__main__':
    unittest.main()
