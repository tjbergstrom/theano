# classifier.py
# Every classifier is able to run batches here


import theano.tensor as T
import theano
import numpy as np
import pickle
from processing import load_test


class Classifier(object):
    def errors(self, x, y):
        raise NotImplementedError


    def pred_label(self, x):
        raise NotImplementedError


    def train(self, dataset, alpha, batch_size, l1_reg, l2_reg, n_epochs):
        raise NotImplementedError


    def run_batches(self, dataset, x, y, train_model_func, batch_size=64, n_epochs=128):
        (train_x, train_y, test_x, test_y, valid_x, valid_y) = dataset
        test_set = load_test(dataset)
        n_train_batches = train_x.get_value(borrow=True).shape[0] // batch_size
        n_valid_batches = valid_x.get_value(borrow=True).shape[0] // batch_size
        n_test_batches = test_x.get_value(borrow=True).shape[0] // batch_size

        idx = T.lscalar()
        test_model = theano.function(
            inputs=[idx],
            outputs=self.errors(x, y),
            givens={
                x: test_x[idx * batch_size:(idx+1)*batch_size],
                y: test_y[idx * batch_size:(idx+1)*batch_size],
            }
        )
        validate_model = theano.function(
            inputs=[idx],
            outputs=self.errors(x, y),
            givens={
                x: valid_x[idx * batch_size:(idx+1)*batch_size],
                y: valid_y[idx * batch_size:(idx+1)*batch_size],
            }
        )

        best_loss = float('inf')
        for epoch in range(n_epochs):
            for minibatch_index in range(n_train_batches):
                train_model_func(minibatch_index)
            validation_loss = [validate_model(i) for i in range(n_valid_batches)]
            avg_validation_loss = np.mean(validation_loss)
            if avg_validation_loss < best_loss:
                best_loss = avg_validation_loss
                test_losses = [test_model(i) for i in range(n_test_batches)]
                avg_test_loss = np.mean(test_losses)
                f = open("best_model.pkl", "w")
                pickle.dump(self, f)
                f.close()
                self.accuracy()
            print(f"Training epoch {epoch}, loss: {avg_validation_loss}")
        return best_loss


    def accuracy(self):
        try:
            f = open("best_model.pickle", "r")
            best_model = pickle.load(f)
            f.close()
            correct = 0
            best_model.batch_size = 1
            for img, label in zip(test_set[0], test_set[1]):
                img = np.asarray(img)
                prediction = best_model.pred_label(img)
                if prediction == label:
                    correct += 1
            accuracy = correct / len(test_set[0])
            print(f"Model accuracy: {accuracy*100}")
        except:
            print(f"Accuracy not available")

##
