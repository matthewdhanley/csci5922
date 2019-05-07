"""
Filter visualization techniques derived from those discussed in
'How to Visualize Convolutional Features in 40 Lines of Code', by Fabio M. Graetz.

https://towardsdatascience.com/how-to-visualize-convolutional-features-in-40-lines-of-code-70b7d87b0030
"""
import torch
from torch.autograd import Variable
import numpy as np
import cv2

from utils.data_loader import load_data


def upscale_image(image, upscale_size):
    """
    Upscales an image with dimensions (3,h,w) to (3,upscale_size,upscale_size).
    """
    image = np.transpose(image, axes=(1, 2, 0))
    image = cv2.resize(image, (upscale_size, upscale_size), interpolation=cv2.INTER_CUBIC)
    image = np.transpose(image, axes=(2, 0, 1))
    return image


class LayerActivations():
    def __init__(self, layer):
        self.forward_hook = layer.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.activations = output

    def remove_hook(self):
        self.forward_hook.remove()


class LayerActivationAnalysis():
    def __init__(self, model, layer):
        self.model = model
        self.layer = layer

    def set_layer(self, layer):
        self.layer = layer

    def get_activated_filter_indices(self, initial_img_size=56):
        """
        Returns a list of indices corresponding to output channels in a given layer
        that were activated by a random input image.
        """
        layer_activations = LayerActivations(self.layer)
        image = (np.random.uniform(0, 255, size=(3, initial_img_size, initial_img_size)) / 255).astype(np.float32,
                                                                                                       copy=False)
        image_tensor = torch.from_numpy(image).expand(1, -1, -1, -1)

        _ = self.model(image_tensor)

        filter_activations = layer_activations.activations[0].detach().numpy()

        layer_activations.remove_hook()

        return np.unique(np.nonzero(filter_activations)[0])


    def get_avg_activated_channels(self, layers, data_path, data_type, sample_size=100):
        '''
        Computes the average number number of channels activated in each layer
        by inputs from the specified dataset.
        '''
        layer_activations = []
        for layer in layers:
            activations = LayerActivations(layer)
            layer_activations.append(activations)

        dataset = load_data(data_path, data_type)
        sampler = torch.utils.data.SubsetRandomSampler(np.arange(sample_size))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, sampler=sampler)

        avg_activated_channels = np.zeros(len(layers))
        for x, _ in dataloader:
            _ = self.model(x)
            channels_activations = [l.activations[0].detach().numpy() for l in layer_activations]
            avg_activated_channels += [len(np.unique(np.nonzero(c)[0])) for c in channels_activations]
        avg_activated_channels = avg_activated_channels / sample_size

        return avg_activated_channels


    def get_max_activating_image(self, channel_index, initial_img_size=56, upscaling_steps=12, upscaling_factor=1.2, lr=0.01, update_steps=15, verbose=False):
        """
        Finds the input image that maximally activates the output channel (with index channel_index)
        of the convolutional layer.
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        layer_activations = LayerActivations(self.layer)
        size = initial_img_size
        image = (np.random.uniform(0, 255, size=(3, size, size)) / 255).astype(np.float32, copy=False)
        for i in range(upscaling_steps):
            image_tensor = torch.from_numpy(image).expand(1, -1, -1, -1)
            image_tensor = Variable(image_tensor, requires_grad=True)

            if not image_tensor.grad is None:
                image_tensor.grad.zero_()

            optimizer = torch.optim.Adam([image_tensor], lr=lr, weight_decay=1e-6)
            image_tensor = image_tensor.to(device)

            # Update image update_steps times
            for n in range(update_steps):
                optimizer.zero_grad()
                _ = self.model(image_tensor)
                loss = -1 * (layer_activations.activations[0, channel_index].norm())
                if verbose and (n % 5 == 0):
                    print('Loss at upscale step {}/{}, update {}/{}: {}'
                          .format(i, upscaling_steps - 1, n, update_steps - 1, loss))
                loss.backward()
                optimizer.step()

            image = torch.squeeze(image_tensor.cpu(), dim=0).clone().detach().numpy()
            size = int(upscaling_factor * size)
            image = upscale_image(image, size)

        output = np.transpose(image, axes=(1, 2, 0))
        layer_activations.remove_hook()
        return np.clip(output, 0, 1)
