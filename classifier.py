# classifier.py


import theano.tensor as T
import theano
import numpy as np
import pickle
from processing import Load_data


class Classifier(object):
    def errors(self, x, y):
        raise NotImplementedError


    def pred_label(self, x):
        raise NotImplementedError


    def train(self, dataset, n_epochs, batch_size, alpha):
        raise NotImplementedError


    def train_batches(self, dataset, x, y, train_model_func, batch_size=64, n_epochs=128):
        (train_x, train_y, test_x, test_y, valid_x, valid_y) = dataset
        test_set = Load_data.testing(dataset)
        n_train_batches = train_x.get_value(borrow=True).shape[0] // batch_size
        n_valid_batches = valid_x.get_value(borrow=True).shape[0] // batch_size
        n_test_batches = test_x.get_value(borrow=True).shape[0] // batch_size

        idx = T.lscalar()
        test_model = theano.function(
            inputs=[idx],
            outputs=self.errors(x, y),
            givens={
                x : test_x[idx * batch_size:(idx+1)*batch_size],
                y : test_y[idx * batch_size:(idx+1)*batch_size],
            }
        )
        validate_model = theano.function(
            inputs=[idx],
            outputs=self.errors(x, y),
            givens={
                x : valid_x[idx * batch_size:(idx+1)*batch_size],
                y : valid_y[idx * batch_size:(idx+1)*batch_size],
            }
        )

        min_loss = float('inf')
        for epoch in range(n_epochs):
            for batch_idx in range(n_train_batches):
                train_model_func(batch_idx)
            validation_loss = [validate_model(i) for i in range(n_valid_batches)]
            avg_validation_loss = np.mean(validation_loss)
            if avg_validation_loss < min_loss:
                min_loss = avg_validation_loss
                test_losses = [test_model(i) for i in range(n_test_batches)]
                avg_test_loss = np.mean(test_losses)
                f = open("model.pickle", "w")
                pickle.dump(self, f)
                f.close()
            print(f"Training epoch {epoch}, Loss: {avg_validation_loss}")
        print(f"Validation Loss: {min_loss}")
        print(f"Test Loss: {avg_test_loss}")
        self.accuracy(test_set)


    def accuracy(self, test_set):
        f = open("model.pickle", "r")
        model = pickle.load(f)
        f.close()
        correct = 0
        model.batch_size = 1
        for img, label in zip(test_set[0], test_set[1]):
            img = np.asarray(img)
            prediction = model.pred_label(img)
            if prediction == label:
                correct += 1
        accuracy = correct / len(test_set[0])
        print(f"Accuracy: {accuracy*100}")



##
