# -*- coding: utf-8 -*-
import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np

def main():
    with open(sys.argv[1], "rb") as file1:
        activs1 = pickle.load(file1)
        file1.close()
    with open(sys.argv[2], "rb") as file2:
        activs2 = pickle.load(file2)
        file2.close()
        
    while True:
        inp = input()
        insplit = inp.split(' ')
        if insplit[0] == 'quit':
            break
        elif insplit[0] == 'view':
            try:
                imnum = int(insplit[1])
                batchnum = int(insplit[2])
                layer = int(insplit[3])
                channel = int(insplit[4])
                im1 = activs1[imnum][layer][batchnum][channel].detach().numpy()
                im2 = activs2[imnum][layer][batchnum][channel].detach().numpy()
                diff = im1 - im2
                plt.subplot(2,2,1)
                plt.imshow(im1)
                plt.subplot(2,2,2)
                plt.imshow(im2)
                plt.subplot(2,2,3)
                plt.imshow(np.maximum(diff,0))
                plt.subplot(2,2,4)
                plt.imshow(np.maximum(-diff,0))
                plt.show()
            except:
                print("Format Error")
            
if __name__ == "__main__":
    main()
    sys.exit(0)