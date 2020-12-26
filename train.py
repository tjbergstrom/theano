# train.py
# Use the convolutional neural network to train an image classification model
#
# $ python3 train.py -d dataset/ -e 32


import sys
import pickle
import random
import argparse
import numpy as np
from cnn import Convolutional
from processing import Load_data


def training(dataset, batch_size=64, epochs=128, alpha=0.01):
    dataset = Load_data.training(dataset)
    num_classes = Load_data.classes(dataset)
    rng = np.random.RandomState(42)
    model = Convolutional(rng=rng, batch_size=batch_size, num_classes)

    model.train(
        dataset,
        alpha=alpha,
        batch_size=batch_size,
        n_epochs=epochs
    )



if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-a", "--alpha", type=int, default=0.01)
    ap.add_argument("-b", "--batch_size", type=int, default=64)
    ap.add_argument("-e", "--num_epochs", type=int, default=128)
    ap.add_argument("-d", "--dataset", type=str, default="dataset/")
    args = vars(ap.parse_args())

    training(args["dataset"], args["batch_size"], args["num_epochs"], args["alpha"])



##
