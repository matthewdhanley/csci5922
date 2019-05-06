# Visualizing Activations in Semantic Segmentation Networks
**Connor Cook, Matt Hanley, Basu Parmar, Galen Pogoncheff, Scott Young**

[![Build Status](https://travis-ci.com/matthewdhanley/csci5922.svg?branch=master)](https://travis-ci.com/matthewdhanley/csci5922)

#### Contents
- [Project Overview](#overview)
- [Comparison Methods](#analysis)
  * [Visualizing Activations](#activ)
  * [Visualizing Maximally Activating Images](#max_activ)
- [Summary of Results](#results)
- [Getting Started](#start)
- [Setup](#setup)
  * [Dependencies](#dependencies)
  * [Project Datasets](#data)
- [Training](#training)
- [Visualizing Activations](#activations)
  * [Save Activations](#activations_save)
  * [Channel Matching](#channel_match)
  * [View Activations](#activations_view)
- [Visualizing Convolutional Channels](#channel_vis)

<a name="overview"></a>
## Project Overview
Neural networks are notoriously known for being difficult to interpret. The visualization techniques proposed by [Zeiler and Fergus (2013)](https://arxiv.org/abs/1311.2901) for visualizing the activity within convolutional networks gave rise to a deeper understanding of what is being learned by convolutional networks, enabling a more methodical model development process. Despite the success of these visualization techniques on convolutional networks designed for image classification, similar approaches are not widely used in networks constructed for semantic segmentation.<br>

This project aims to address this problem by comparing visualizations from networks trained for image classification and networks trained for semantic segmentation.  These comparisions were then used to conclude whether the task of semantic segmentaion leads the encoder portion of the network to learn inherently different features than it would if the network to be trained for image classification.<br>

The two networks that we used for this analysis were a VGG-11 classification network pretrained on the imagenet dataset and an implementation of [TernausNet](https://arxiv.org/abs/1801.05746), a U-Net which features a VGG-11 encoder.  We trained the U-Net on the [Cityscapes](https://www.cityscapes-dataset.com) semantic segmentation dataset.

<a name="analysis"></a>
## Comparison Methods

<a name="activ"></a>
## Visualizing Activations
<a name="max_activ"></a>
## Visualizing Maximally Activating Images

<a name="results"></a>
## Summary of Results

<a name="start"></a>
## Getting Started

<a name="setup"></a>
## Setup

<a name="dependencies"></a>
### Dependencies
This source code for this project was written in Python 3.  The following command will install the required project dependecies.
```
pip install -r requirements.txt
```

<a name="data"></a>
### Project Datasets
We used the [Cityscapes](https://www.cityscapes-dataset.com/downloads/) dataset for semantic segmentaion (fine annotations) to train our U-Net model.  Network activation visualizations were obtained using images from Cityscapes dataset and the [Tiny ImageNet](https://tiny-imagenet.herokuapp.com/) dataset.


#### Cityscapes dataset directory structure
Shown below is the Cityscape data hierarchy expectedby our training script.  Within each of the test, train, and val folders are folders named after the city that the data was taken. These folders hold the respective data.
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

<a name="training"></a>
## Training
```
python main.py cityscapes_path/ --mode train --save_dir checkpoints/ --file unet.tar
```

<a name="activations"></a>
## Visualizing Activations

<a name="activations_save"></a>
#### Save Activations
```
python main.py path --mode ...
```

<a name="channel_match"></a>
#### Channel Matching
```
python main.py path --mode ...
```

<a name="activations_view"></a>
#### View Activations
```
python main.py path --mode ...
```

<a name="channel_vis"></a>
## Visualizing Convolutional Channels
```
python main.py path --mode ...
```
