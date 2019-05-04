import sys
import os
import torch
from torch import nn
from torchvision import datasets

from utils.args import get_cli_arguments
from utils.data_loader import load_data
from models.UNet import UNet
from models.VGGmod import VGGmod
from train import train
from test import test
from match_channels import match_channels
from retrieve_activations import retrieve_activations
from utils.set_parameter_required_grad import set_parameter_required_grad
from utils.load_activations import load_activations
from view_activs import visualize_batch
import numpy as np


def main():
    args = get_cli_arguments()

    if (args.checkpoint is not None) and (not os.path.exists(args.checkpoint)):
        sys.exit('Specified checkpoint cannot be found')

    if args.mode == 'train':
        if args.dataset != 'cityscapes':
            sys.exit("Model can only be trained on cityscapes dataset")

        dataset = load_data(args.path, args.dataset, resize=~args.no_resize)

        if args.subset:
            sampler = torch.utils.data.SubsetRandomSampler(np.arange(10))
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)
        else:
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        model = UNet(num_classes=len(datasets.Cityscapes.classes), pretrained=args.pretrained)

        if not args.pretrained:
            set_parameter_required_grad(model, True)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        if (args.savedir is not None) and (not os.path.exists(args.savedir)):
            os.makedirs(args.savedir)
        train(model, dataloader, criterion, optimizer, num_epochs=args.epochs, checkpoint_path=args.checkpoint,
              save_path=args.savedir)
        return

    if args.mode == 'test':
        dataset = load_data(args.path, args.dataset, resize=~args.no_resize, split='val')

        if args.subset:
            sampler = torch.utils.data.SubsetRandomSampler(np.arange(10))
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)
        else:
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        model = UNet(num_classes=len(datasets.Cityscapes.classes), pretrained=args.pretrained)
        test(model, dataloader, args.checkpoint)

    if args.mode == 'activations':
        if args.model is None:
            sys.exit("Must specify model to use with --model argument")
        dataset = load_data(args.path, args.dataset, resize=~args.no_resize)
        if args.subset:
            sampler = torch.utils.data.SubsetRandomSampler(np.arange(50))
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)
        else:
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        if args.model == 'unet':
            model = UNet(num_classes=len(datasets.Cityscapes.classes))
            if args.checkpoint:
                checkpoint = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                print("NOTE: Getting activations for untrained network. Specified a pretrained model with the "
                      "--checkpoint argument.")
        elif args.model == 'vggmod':
            model = VGGmod()
        else:
            model = UNet(num_classes=len(datasets.Cityscapes.classes))
            if args.checkpoint:
                checkpoint = torch.load(args.checkpoint)
                model.load_state_dict(checkpoint['model_state_dict'])
            set_parameter_required_grad(model, True)
            retrieve_activations(model, dataloader, args.dataset)
            model = VGGmod()

        set_parameter_required_grad(model, True)

        retrieve_activations(model, dataloader, args.dataset)

    if args.mode == 'view_activations':
        file_1 = os.path.join(args.path, 'VGGmod_activations')
        file_2 = os.path.join(args.path, 'UNet_activations_matched')
        if not os.path.exists(file_1) or not os.path.exists(file_2):
            exit("Could not load activations from "+args.path+". If you have not generated activations for both UNet "
                 "and VGG11, run instead with the \"--mode activations\" parameter.")

        activs1, activs2 = load_activations(file_1, file_2)
        visualize_batch(activs1, activs2,
                        batch_num=args.batch_num,
                        start_layer=args.start_layer,
                        stop_layer=args.stop_layer)

    if args.mode == 'compare_activations':
        file_1 = os.path.join(args.path, 'VGGmod_activations')
        file_2 = os.path.join(args.path, 'UNet_activations')
        match_channels(file_1, file_2, args.type)


if __name__ == "__main__":
    main()
    sys.exit(0)
