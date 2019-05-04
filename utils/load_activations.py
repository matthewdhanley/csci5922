import pickle


def load_activations(f1, f2):
    """
    Loads activations from given file
    :param f1: file holding pickle activations
    :param f2: file holding pickle activations
    :return: activations from f1, activations from f2
    """
    with open(f1, "rb") as file1:
        activs1 = pickle.load(file1)
        file1.close()
    with open(f2, "rb") as file2:
        activs2 = pickle.load(file2)
        file2.close()

    return activs1, activs2
