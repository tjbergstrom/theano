# mlp.py
# Convolutional layers


import theano
import theano.tensor as T
import numpy as np


class Pooling(object):
    def __init__(self, rng, filter_shape, img_shape, poolsize=(2,2)):
        if filter_shape[1] != img_shape[1]:
            raise TypeError(f"Expected shape {filter_shape[1]}, got {img_shape[1]}")
        fin = np.prod(filter_shape[1:])
        fout = filter_shape[0] * np.prod(filter_shape[2:]) // np.prod(poolsize)
        W_bound = np.sqrt(6.0 / (fin + fout))
        self.W = theano.shared(
            value=np.asarray(
                rng.uniform(low=(-W_bound), high=W_bound, size=filter_shape),
                dtype=theano.config.floatX,
            ),
            borrow=True,
        )
        b_values = np.zeros(shape=(filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)
        self.filter_shape = filter_shape
        self.img_shape = img_shape
        self.poolsize = poolsize
        self.params = [self.W, self.b]


    def output(self, input_layer):
        conv_out = conv2d(
            input=input_layer,
            filters=self.W,
            filter_shape=self.filter_shape,
            input_shape=self.img_shape
        )
        pool_out = pool.pool_2d(
            input=conv_out,
            ds=self.poolsize,
            ignore_border=True,
        )
        lin_output = pool_out + self.b.dimshuffle('x', 0, 'x', 'x')
        return T.tanh(lin_output)



class Hidden(object):
    def __init__(self, n_in, n_out, rng, activation=T.tanh):
        self.W = theano.shared(
            value=np.asarray(
                rng.uniform(
                    low=-np.sqrt(6.0 / (n_in+n_out)),
                    high=np.sqrt(6.0 / (n_in+n_out)),
                    size=(n_in, n_out),
                ),
                dtype=theano.config.floatX,
            ),
            name='W', borrow=True,
        )
        self.b = theano.shared(
            value=np.zeros((n_out,), dtype=theano.config.floatX),
            name='b', borrow=True,
        )
        self.activation = activation
        self.params = [self.W, self.b]


    def output(self, in_vect):
        lin_output = T.dot(in_vect, self.W) + self.b
        return self.activation(lin_output)



##
