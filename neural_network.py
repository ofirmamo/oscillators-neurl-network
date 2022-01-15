from typing import List

import numpy as np

from data_loader import DataLoader
from utilities.activation_functions import FUNC, DERIVATIVE
from utilities.constants import *


# Depth: 2
# Width:
#   First layer: 1024
#   Second layer: 512

# Runs on epochs, CNN Neural network
# Parameters:
#   Data Loader, iterable each iteration is a batch.
#   Activation Function.
#   Error Function
#   Learning Rate -
class CNNNeuralNetwork:

    def __init__(self, activation_func, learning_rate):
        self.activation_func = activation_func
        self.learning_rate = learning_rate

        np.random.seed(RANDOM_SEED)
        self.synapse_0 = 2 * np.random.random((WIDTH_LAYER_1, WIDTH_LAYER_2)) - 1
        self.synapse_1 = 2 * np.random.random((WIDTH_LAYER_2, 1)) - 1

    def train_return_acc_and_loss(self, epochs, print_every=25):
        data_loader = DataLoader()
        acc = []
        loss = []
        test_acc = []
        test_loss = []
        weights = []

        for e in range(epochs):
            weights.append((np.copy(self.synapse_0), np.copy(self.synapse_1)))
            epoch_loss = self.run_epoch(data_loader)
            avg_epoch_loss = epoch_loss / len(data_loader)
            avg_epoch_acc = (1 - avg_epoch_loss) * 100

            epoch_test_acc, epoch_test_loss = self.predict_samples_return_acc_and_loss(
                data_loader.get_samples_with_expected_result(data_loader.test_set))
            test_loss.append(epoch_test_loss)
            test_acc.append(epoch_test_acc)

            if e % print_every == 0:
                print(f"**[{e + 1} Epoch]**")
                print(f"avg_acc = {avg_epoch_acc}")
                print(f"avg_loss = {avg_epoch_loss}")
                print(f"test_acc = {epoch_test_acc}")
                print(f"test_loss = {epoch_test_loss}")
            acc.append(avg_epoch_acc)
            loss.append(avg_epoch_loss)

        return acc, loss, test_acc, test_loss, weights

    def run_epoch(self, data_loader: DataLoader):
        epoch_batch_loss = 0
        for _, batch_samples, batch_expected in data_loader:
            # Feed forward
            layer_1_output = self.activation_func[FUNC](np.dot(batch_samples, self.synapse_0))
            batch_prediction = self.activation_func[FUNC](np.dot(layer_1_output, self.synapse_1))

            # errors
            batch_output_error = batch_expected - batch_prediction
            batch_output_delta = batch_output_error * self.activation_func[DERIVATIVE](batch_prediction)

            # collect stats
            epoch_batch_loss += np.sum(np.power(batch_output_error, 2))

            # Backpropagation
            layer_1_error = batch_output_delta.dot(self.synapse_1.T)
            layer_1_delta = layer_1_error * self.activation_func[DERIVATIVE](layer_1_output)
            # update weights (synapses)
            self.synapse_1 += self.learning_rate * layer_1_output.T.dot(batch_output_delta)
            self.synapse_0 += self.learning_rate * batch_prediction.T.dot(layer_1_delta)

        # for stats
        return epoch_batch_loss

    def predict_samples(self, samples_and_expected, synapse0=None, synapse1=None):
        synapse0 = self.synapse_0 if synapse0 is None else synapse0
        synapse1 = self.synapse_1 if synapse1 is None else synapse1

        samples, expected = samples_and_expected
        layer_1_output = self.activation_func[FUNC](np.dot(samples, synapse0))
        results = self.activation_func[FUNC](np.dot(layer_1_output, synapse1))

        return results

    def predict_samples_return_acc_and_loss(self, files_samples_and_expected, synapse0=None, synapse1=None):
        synapse0 = self.synapse_0 if synapse0 is None else synapse0
        synapse1 = self.synapse_1 if synapse1 is None else synapse1

        _, samples, expected = files_samples_and_expected
        layer_1_output = self.activation_func[FUNC](np.dot(samples, synapse0))
        results = self.activation_func[FUNC](np.dot(layer_1_output, synapse1))
        loss = np.sum(np.power(expected - results, 2)) / len(results)
        acc = (1 - loss) * 100
        return acc, loss
