# train.py
# Use the convolutional neural network to train an image classification model
#
# $ python3 train.py -d dataset/ -e 32


import sys
import pickle
import random
import argparse
import numpy as np
from mlp import MLP
from cnn import Convolutional
from processing import load_data


def training(dataset, batch_size=64, epochs=128, alpha=0.01):
    dataset = load_data(dataset)
    rng = np.random.RandomState(42)
    classifier = Convolutional(rng=rng, batch_size=batch_size, nkerns=(20, 50))

    classifier.train(
        dataset,
        alpha=alpha,
        l1_reg=0.00001,
        l2_reg=0.001,
        batch_size=batch_size,
        n_epochs=epochs
    )

    classifier.accuracy()



if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-a", "--alpha", type=int, default=0.01)
    ap.add_argument("-b", "--batch_size", type=int, default=64)
    ap.add_argument("-e", "--num_epochs", type=int, default=128)
    ap.add_argument("-d", "--dataset", type=str, default="dataset/")
    args = vars(ap.parse_args())

    training(args["dataset"], args["batch_size"], args["num_epochs"], args["alpha"])



##
