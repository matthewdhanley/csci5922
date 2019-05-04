# CSCI 5922 - Deep Learning Final Project

This project is an implementation of [TernausNet](https://arxiv.org/abs/1801.05746), a U-Net with a VGG11 encoder. The
network used in TerausNet made use of a pre-trained (on imagenet) VGG11 encoder. 

The goal of this project is to determine how activations differ between an encoder trained as a classifier on a dataset
such as ImageNet and that same encoder that is trained as part of a full segmentation network.

We used the [Cityscapes](https://www.cityscapes-dataset.com) dataset for the training of this network.

[![Build Status](https://travis-ci.com/matthewdhanley/csci5922.svg?branch=master)](https://travis-ci.com/matthewdhanley/csci5922)

### Training
```
python main.py cityscapes_path/ --mode train --save_dir checkpoints/ --file unet.tar
```


### Cityscapes dataset directory structure
Within each of the test, train, and val folders are folders named after the city that the data was taken. These folders 
hold the respective data.
```
.
├── gtFine
│   ├── test
│   ├── train
│   └── val
└── leftImg8bit
    ├── test
    ├── train
    └── val
```