# -*- coding: utf-8 -*-
import pickle
import numpy as np
import torch
import copy
from skimage.measure import _structural_similarity as ssim


def load_files(fin1, fin2):
    with open(fin1, mode='rb') as f1:
        with open(fin2, mode='rb') as f2:
            activations_1 = pickle.load(f1)
            activations_2 = pickle.load(f2)
    return activations_1, activations_2


def match_channels(fin1, fin2):
    activations_1, activations_2 = load_files(fin1, fin2)
    num_images = len(activations_1)
    num_layers = len(activations_1[0])
    activs_out = copy.deepcopy(activations_1)
    for i in range(num_layers):
        print("Layer {}".format(i))
        num_channels = len(activations_1[0][i][0, :, 0, 0])
        dot_cum = np.zeros((num_channels, num_channels))
        mag_cum1 = np.zeros((num_channels, num_channels))
        mag_cum2 = np.zeros((num_channels, num_channels))

        for j in range(num_images):
            for k in range(num_channels):
                for l in range(num_channels):
                    dot_cum[k, l] += torch.sum(torch.mul(activations_1[j][i][:, k, :, :],
                                                         activations_2[j][i][:, l, :, :]))
                    mag_cum1[k, l] += torch.sum(torch.mul(activations_1[j][i][:, k, :, :],
                                                          activations_1[j][i][:, k, :, :]))
                    mag_cum2[k, l] += torch.sum(torch.mul(activations_2[j][i][:, l, :, :],
                                                          activations_2[j][i][:, l, :, :]))

        root = (np.sqrt(mag_cum1 * mag_cum2))
        root[root <= .00001] = float('Inf')
        scores = dot_cum / root
        # for j in range(num_images):
        #     activs_out[j][i] = torch.zeros_like(activations_1[j][i])
        # print(activations_1[0][0].shape)
        # print(activs_out[0][0].shape)

        while np.max(scores) != -float('Inf'):
            ind = np.unravel_index(np.argmax(scores, axis=None), scores.shape)
            for j in range(num_images):
                # print(j)
                # print(i)
                # print(activs_out[j][i].shape)
                # print(activations_1[j][i].shape)
                activs_out[j][i][:, ind[1], :, :] = activations_1[j][i][:, ind[0], :, :]
            scores[ind[0], :] = -float('Inf')
            scores[:, ind[1]] = -float('Inf')
    with open(fin2 + '_matched', mode='wb') as fout:
        pickle.dump(activs_out, fout)



