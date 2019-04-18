import sys
import os
import torch
from torch import nn
from torchvision import datasets

from utils.args import get_cli_arguments
from utils.data_loader import load_data
from models.UNet import UNet
from train import train
import numpy as np


def main():
    args = get_cli_arguments()

    if (args.checkpoint is not None) and (not os.path.exists(args.checkpoint)):
        sys.exit('Specified checkpoint cannot be found')

    if args.mode == 'train':
        dataset = load_data(args.path, resize=~args.no_resize)

        if args.subset:
            sampler = torch.utils.data.SubsetRandomSampler(np.arange(10))
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)
        else:
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        model = UNet(num_classes=len(datasets.Cityscapes.classes))

        set_parameter_required_grad(model, True)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        train(model, dataloader, criterion, optimizer, num_epochs=args.epochs, checkpoint_path=args.checkpoint)
        return


def set_parameter_required_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad
    return


if __name__ == "__main__":
    main()
    sys.exit(0)
