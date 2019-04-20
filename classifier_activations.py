import sys
import torch
from models.VGGmod import VGGmod
from utils.data_loader import load_data
import pickle


def main():
    net = VGGmod()
    dataset = load_data("Cityscapes/")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    activs = []
    for inp, _ in dataloader:
        net(inp)
        activs.append(net.activs)
    file = open("classifier_activations", "wb")
    pickle.dump(activs, file)
    file.close()
    
    
if __name__ == '__main__':
    main()
    sys.exit(0)