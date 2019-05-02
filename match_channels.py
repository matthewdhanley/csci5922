# -*- coding: utf-8 -*-
import pickle
import numpy as np
import copy


def load_files(fin1, fin2):
    with open(fin1, mode='rb') as f1:
        with open(fin2, mode='rb') as f2:
            activations_1 = pickle.load(f1)
            activations_2 = pickle.load(f2)
    return activations_1, activations_2


def match_channels(fin1, fin2, mode):
    '''
    Matches the channels of in each layer of two encoder modules based
    on cosine similarity of their layer activations.
    
    Params:
    fin1: pickle file containing activations from encoder 1
    fin2: pickle file containing activations from encoder 2
    mode: Preprocessing method to be applied to the activations
    prior to their comparison.  Applied in effort to reduce the
    pixel locality dependency in cosine similarity.
    '''
    if not mode in ['normal', 'blur', 'dilation', 'pooling']:
        print(mode + " is not a valid matching type.")
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
            activs_1 = activations_1[j][i].detach().numpy()
            activs_2 = activations_2[j][i].detach().numpy()
            if mode == 'blur':
                dim = list(activs_1.shape)
                dim[2] -= 2
                dim[3] -= 2
                mod_activs_1 = np.zeros(dim)
                mod_activs_2 = np.zeros(dim)
                for k in range(dim[2]):
                    for l in range(dim[3]):
                        mod_activs_1[:,:,k,l] = np.mean(activs_1[:,:,k:(k+2),l:(l+2)], (2,3))
                        mod_activs_2[:,:,k,l] = np.mean(activs_2[:,:,k:(k+2),l:(l+2)], (2,3))
            elif mode == 'dilation':
                dim = list(activs_1.shape)
                dim[2] -= 2
                dim[3] -= 2
                mod_activs_1 = np.zeros(dim)
                mod_activs_2 = np.zeros(dim)
                for k in range(dim[2]):
                    for l in range(dim[3]):
                        mod_activs_1[:,:,k,l] = np.max(activs_1[:,:,k:(k+2),l:(l+2)], (2,3))
                        mod_activs_2[:,:,k,l] = np.max(activs_2[:,:,k:(k+2),l:(l+2)], (2,3))
            elif mode == 'pooling':
                dim = list(activs_1.shape)
                dim[2] = int(np.floor((dim[2]-5)/3))
                dim[3] = int(np.floor((dim[3]-5)/3))
                mod_activs_1 = np.zeros(dim)
                mod_activs_2 = np.zeros(dim)
                for k in range(dim[2]):
                    for l in range(dim[3]):
                        mod_activs_1[:,:,k,l] = np.max(activs_1[:,:,(3*k):((3*k)+5),(3*l):((3*l)+5)], (2,3))
                        mod_activs_2[:,:,k,l] = np.max(activs_2[:,:,(3*k):((3*k)+5),(3*l):((3*l)+5)], (2,3))
            elif mode == 'normal':
                mod_activs_1 = activs_1
                mod_activs_2 = activs_2
            for k in range(num_channels):
                for l in range(num_channels):
                    dot_cum[k, l] += np.sum(np.multiply(mod_activs_1[:, k, :, :],
                                                         mod_activs_2[:, l, :, :]))
                    mag_cum1[k, l] += np.sum(np.multiply(mod_activs_1[:, k, :, :],
                                                          mod_activs_1[:, k, :, :]))
                    mag_cum2[k, l] += np.sum(np.multiply(mod_activs_2[:, l, :, :],
                                                          mod_activs_2[:, l, :, :]))

        root = (np.sqrt(mag_cum1 * mag_cum2))
        root[root <= .00001] = float('Inf')
        scores = dot_cum / root

        while np.max(scores) != -float('Inf'):
            ind = np.unravel_index(np.argmax(scores, axis=None), scores.shape)
            for j in range(num_images):
                activs_out[j][i][:, ind[1], :, :] = activations_1[j][i][:, ind[0], :, :]
            scores[ind[0], :] = -float('Inf')
            scores[:, ind[1]] = -float('Inf')
    with open(fin2 + '_matched', mode='wb') as fout:
        pickle.dump(activs_out, fout)
