import pickle
import os


def retrieve_activations(net, dataloader, dataset_name):
    activs = []
    if not os.path.exists('activations'):
        print("Creating directory ./activations to store activations.")
        os.makedirs('activations')
    model_name = net.name
    for inp, _ in dataloader:
        net(inp)
        activs.append(net.activs)
    file = open("activations/{}_activations".format(model_name), "wb")
    print("Saving activations to activations/{}_activations".format(model_name))
    pickle.dump(activs, file)
    file.close()
