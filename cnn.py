

from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool
import theano
import theano.tensor as T
import numpy as np
from classifier import Classifier
from log_reg import Logistic_Regression
from layers import Hidden
from layers import Pooling


class Convolutional(Classifier):
    def __init__(self, rng, batch_size, num_classes, k=3, dense=512, nkerns=(20,50)):
        self.batch_size = batch_size
        # 28x28 pooling layer
        self.layer0 = Pooling(
            rng=rng,
            image_shape=(batch_size, 1, 28, 28),
            filter_shape=(nkerns[0], 1, k, k),
        )
        # 12x12 pooling layer
        self.layer1 = Pooling(
            rng=rng,
            image_shape=(batch_size, nkerns[0], 12, 12),
            filter_shape=(nkerns[1], nkerns[0], k, k)
        )
        # 4x4 hidden layer
        self.layer2 = Hidden(
            rng=rng,
            n_in=nkerns[1]*4*4,
            n_out=dense,
            activation=T.tanh,
        )
        # Output layer/prediction
        self.layer3 = Logistic_Regression(
            n_in=dense,
            n_out=num_classes,
        )


    def layers_outputs(self, x):
        layer0_input = x.reshape((self.batch_size, 1, 28, 28))
        l0_output = self.layer0.output(layer0_input)
        l1_output = self.layer1.output(l0_output)
        l2_input = l1_output.flatten(2)
        l2_output = self.layer2.output(l2_input)
        return l2_output


    def negative_log_likelihood(self, x, y):
        output = self.layers_outputs(x)
        return self.layer3.negative_log_likelihood(output, y)


    def pred_label(self, x):
        output = self.layers_outputs(x)
        output = output.flatten(1)
        return self.layer3.pred_label(output)


    def errors(self, x, y):
        output = self.layers_outputs(x)
        return self.layer3.errors(output, y)


    def train(self, dataset, n_epochs=128, batch_size=64, alpha=0.10):
        (train_x, train_y, test_x, test_y, valid_x, valid_y) = dataset
        x = T.matrix('x')
        y = T.ivector('y')
        batch_size = self.batch_size

        layer0_input = x.reshape((batch_size, 1, 28, 28))
        cost = self.negative_log_likelihood(layer0_input, y)
        params = self.layer0.params+self.layer1.params+self.layer2.params+self.layer3.params
        grads = T.grad(cost, params)
        updates = [(param, param-alpha * grad) for param, grad in zip(params, grads)]

        idx = T.lscalar()
        train_func = theano.function(
            inputs=[idx],
            outputs=cost,
            updates=updates,
            givens={
                x: train_x[idx * batch_size:(idx+1)*batch_size],
                y: train_y[idx * batch_size:(idx+1)*batch_size],
            }
        )
        self.train_batches(
            dataset, x, y,
            train_model_func=train_func,
            batch_size=batch_size,
            n_epochs=n_epochs
        )



##
