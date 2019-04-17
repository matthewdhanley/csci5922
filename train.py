import torch
import torch.nn as nn
from torchvision import transforms, datasets
from UNet import UNet
from PIL import Image
from data_transforms import PILToLongTensor
import numpy as np
import os
import sys
import argparse
import time





def train(model, dataloader, criterion, optimizer, num_epochs, ckpt_path=None):
    since = time.time()
    min_loss = float('inf')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)

    if ckpt_path is not None:
        print('Resuming training from ckeckpoint...')
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        epoch = ckpt['epoch']
        min_loss = ckpt['loss']

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 11)

        running_loss = 0.0

        # Iterate over data.
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)

        print('Training Loss: {}'.format(epoch_loss))

        if epoch_loss < min_loss:
            min_loss = epoch_loss

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss
            }, './checkpoints/unet.tar')

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return model


def load_data(path):
    input_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    output_transform = transforms.Compose([
        transforms.Resize((256, 256), Image.NEAREST),
        PILToLongTensor()
    ])

    dataset = datasets.Cityscapes(path, split='train', mode='fine',
                                  target_type='semantic', transform=input_transform,
                                  target_transform=output_transform)

    return dataset



def set_parameter_required_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad
    return


def main():
    parser = argparse.ArgumentParser(description='Train UNet on CityScapes data')
    parser.add_argument('path', type=str,
                        help='Relative path to directory containing to CityScapes gtFine and leftImg8bit directories')
    parser.add_argument('--ckpt', type=str, default=None, help='Relative path to saved checkpoint')
    parser.add_argument('-b', '--batch_size', type=int, default=8, help='Image batch size (default 0.001)')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='Learning rate (default 0.001)')
    parser.add_argument('--epochs', type=int, default=100, help='Maximum number of training epochs')
    args = parser.parse_args()

    if (args.ckpt is not None) and (not os.path.exists(args.ckpt)):
        sys.exit('Specified checkpoint cannot be found')

    dataset = load_data(args.path)
    sampler = torch.utils.data.SubsetRandomSampler(np.arange(10))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = UNet(num_classes=len(datasets.Cityscapes.classes))
    set_parameter_required_grad(model, True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    train(model, dataloader, criterion, optimizer, num_epochs=args.epochs, ckpt_path=args.ckpt)
    return


if __name__ == '__main__':
    main()
    sys.exit(0)
