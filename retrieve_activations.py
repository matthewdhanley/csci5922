import pickle
import os


def retrieve_activations(net, dataloader):
    activs = []
    for inp, _ in dataloader:
        net(inp)
        activs.append(net.activs)
    model_name = net.name
    if not os.path.exists('activations'):
        print("Creating directory ./activations to store activations.")
        os.makedirs('activations')
    print("Saving activations to activations/{}_activations".format(model_name))
    file = open("activations/{}_activations".format(model_name), "wb")
    pickle.dump(activs, file)
    file.close()
