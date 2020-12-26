# log_reg.py
# Logistic regression functions for classifiers


import theano
import theano.tensor as T
import numpy as np
from classifier import Classifier


class Logistic_Regression(Classifier):
    def __init__(self, n_in, n_out):
        self.W = theano.shared(
            value=np.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX,
            ),
            name='W', borrow=True,
        )
        self.b = theano.shared(
            value=np.zeros(
                (n_out,),
                dtype=theano.config.floatX,
            ),
            name='b', borrow=True,
        )
        self.p_y_given_x = T.nnet.softmax(T.dot(x, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_x, axis=1)
        self.params = [self.W, self.b]


    def p_y_x(self, in_vect):
        return T.nnet.softmax(T.dot(in_vect, self.W) + self.b)


    def predict(self, x):
        return T.argmax(self.p_y_x(x), axis=1)


    def negative_log_likelihood(self, x, y):
        samples_in_batch = y.shape[0]
        log_probs = T.log(self.p_y_x(x))
        log_likelihoods = log_probs[(T.arange(samples_in_batch), y)]
        return -T.mean(log_likelihoods)


    def pred_label(self, x):
        x_ = T.vector("x")
        label = theano.function(
            inputs=[],
            outputs=self.predict(x_),
            givens={
                x_ : x
            }
        )()[0]
        return label


    def errors(self, x, y):
        y_pred = self.predict(x)
        if y.ndim != y_pred.ndim:
            raise TypeError(f"Expected {y_pred.ndim}, got {y.ndim}")
        if not y.dtype.startswith("int"):
            raise TypeError()
        equalities = T.neq(y, y_pred)
        return T.mean(equalities)



##
