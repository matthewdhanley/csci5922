# -*- coding: utf-8 -*-
import sys
import pickle
import numpy as np
import torch


def main():
    file1in = open(sys.argv[1], "rb")
    activs1 = pickle.load(file1in)
    file1in.close()
    file2in = open(sys.argv[2], "rb")
    activs2 = pickle.load(file2in)
    file2in.close()
    
    
    
    num_images = len(activs1)
    num_layers = len(activs1[0])
    
    
    activs_out = num_images*[num_layers*[None]]
    
    for i in range(num_layers):
        num_channels = len(activs1[0][i][0,:,0,0])
        dot_cum = np.zeros((num_channels,num_channels))
        mag_cum1 = np.zeros((num_channels,num_channels))
        mag_cum2 = np.zeros((num_channels,num_channels))



        for j in range(num_images):
            for k in range(num_channels):
                for l in range(num_channels):
                    dot_cum[k,l] += torch.sum(torch.mul(activs1[j][i][:,k,:,:],activs2[j][i][:,l,:,:]))
                    mag_cum1[k,l] += torch.sum(torch.mul(activs1[j][i][:,k,:,:],activs1[j][i][:,k,:,:]))
                    mag_cum2[k,l] += torch.sum(torch.mul(activs2[j][i][:,l,:,:],activs2[j][i][:,l,:,:]))
                    
        root = (np.sqrt(mag_cum1 * mag_cum2))
        root[root <= .00001] = float('Inf')
        scores = dot_cum / root
        
        print(scores)
        for j in range(num_images):
            activs_out[j][i] = torch.zeros(activs1[j][i].size())
            
        while (np.max(scores) != -float('Inf')):
            ind = np.unravel_index(np.argmax(scores, axis=None), scores.shape)
            for j in range(num_images):
                activs_out[j][i][:,ind[1],:,:] = activs1[j][i][:,ind[0],:,:]
            scores[ind[0],:] = -float('Inf')
            scores[:,ind[1]] = -float('Inf')
            
    file2out = open(sys.argv[2], "wb")
    pickle.dump(activs_out, file2out)
    file2out.close()
        

    
    
if __name__ == '__main__':
    main()