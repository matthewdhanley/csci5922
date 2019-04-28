# -*- coding: utf-8 -*-
import sys
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize


def visualize_batch(activations1, activations2, batch_num, dpi=300, alpha=0.5, cmap="hot"):
    """
    Visualize the activations in activations list. Activations will be saved to subfolders in activation_visualizations/.
    If this directory is not present, it will be created automatically.
    :param activations1: Full list of activations
    :param activations2: Full list of actiations
    :param batch_num: Batch number to visualize.
    :param dpi: dpi used to save figure. Default 300
    :param alpha: Alpha of channel overlay used. Default 0.5
    :param cmap: cmap of overlay used. Default "hot"
    :return: None
    """
    if not os.path.exists('activation_visualizations'):
        os.makedirs('activation_visualizations')

    for i in range(len(activations1)):
        savedir_base = 'activation_visualizations/img{}_batch{}'.format(i, batch_num)
        if not os.path.exists(savedir_base):
            os.makedirs(savedir_base)
        for j in range(len(activations1[i])):

            savedir = os.path.join(savedir_base, 'layer{}'.format(j))
            if not os.path.exists(savedir):
                os.makedirs(savedir)
            for l in range(len(activations1[i][j][batch_num])):
                im1 = activations1[i][j][batch_num][l].detach().numpy()
                im2 = activations2[i][j][batch_num][l].detach().numpy()

                original = activations1[i][0][batch_num][:].detach().numpy()

                diff = im1 - im2
                fig, axs = plt.subplots(3, 2)
                fig.suptitle("Batch {} | Image {} | Layer {} | Channel {}".format(batch_num, i, j, l), fontsize=16)

                ax = plt.subplot(3, 2, 1)
                ax.set_title(sys.argv[1].split('/')[-1])

                plt.imshow(im1, cmap="gray")

                ax = plt.subplot(3, 2, 2)
                ax.set_title(sys.argv[2].split('/')[-1])
                plt.imshow(im2, cmap="gray")
                plt.axis("off")

                plt.subplot(3, 2, 3)
                plt.imshow(np.rollaxis(original, 0, 3))

                if original.shape != im1.shape:
                    im1 = resize(im1, original[0].shape, anti_aliasing=True)
                    im2 = resize(im2, original[0].shape, anti_aliasing=True)

                plt.imshow(im1, cmap=cmap, alpha=alpha)
                plt.axis("off")

                plt.subplot(3, 2, 4)
                plt.imshow(np.rollaxis(original, 0, 3))

                plt.imshow(im2, cmap=cmap, alpha=alpha)
                plt.axis("off")

                plt.subplot(3, 2, 5)
                plt.imshow(np.maximum(diff, 0))
                plt.axis("off")

                plt.subplot(3, 2, 6)
                plt.imshow(np.maximum(-diff, 0))
                plt.axis("off")

                fname = os.path.join(savedir, 'im{}_batch{}_layer{}_cha{}.png'.format(i, batch_num, j, l))
                fig.savefig(fname, dpi=dpi)
                plt.close(fig)


def main():
    with open(sys.argv[1], "rb") as file1:
        activs1 = pickle.load(file1)
        file1.close()
    with open(sys.argv[2], "rb") as file2:
        activs2 = pickle.load(file2)
        file2.close()

    visualize_batch(activs1, activs2, 0)


if __name__ == "__main__":
    main()
    sys.exit(0)