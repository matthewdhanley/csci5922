import torch
import time
import datetime as dt


def train(model, dataloader, criterion, optimizer, num_epochs, checkpoint_path=None):
    since = time.time()
    min_loss = float('inf')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Training Device: {}".format(device))
    model.to(device)

    if checkpoint_path is not None:
        print('Resuming training from ckeckpoint...')
        ckpt = torch.load(checkpoint_path)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        epoch = ckpt['epoch']
        min_loss = ckpt['loss']

    for epoch in range(num_epochs):
        print('{} -- Epoch {}/{}'.format(dt.datetime.now(), epoch, num_epochs - 1))
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

        if epoch_loss < min_loss and checkpoint_path is not None:
            min_loss = epoch_loss

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, './' + checkpoint_path + '/unet.tar')

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return model

