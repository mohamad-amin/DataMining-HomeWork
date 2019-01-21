import numpy as np


def tanh(x):
    return np.tanh(x)


def tanh_derivation(x):
    return 1 - tanh(x)**2


def softmax(x):
    exp = np.exp(x)
    return exp / np.sum(exp)


def softmax_derivation(x):
    raise NotImplementedError("You shouldn'nt be using this, you're doing something wrong!")
