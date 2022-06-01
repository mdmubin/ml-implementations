import numpy as np


def sigmoid(x):
    def _sig(i): return 1 / (1 + np.exp(-i))
    return np.array([_sig(value) for value in x])


def loss(original, predicted):
    return (- original * np.log(predicted + 1e-9) - (1 - original) * np.log(1 - predicted + 1e-9))


def gradient_descent(x, original, predicted):
    diff = predicted - original
    db = np.mean(diff)
    dw = np.matmul(x.transpose(), diff)
    dw = np.array([np.mean(i) for i in dw])
    return dw, db


def normalize(x):
    for i in range(x.shape[1]):
        x[:, i] = (x[:, i] - np.mean(x[:, i]))/np.std(x[:, i])
    return x
