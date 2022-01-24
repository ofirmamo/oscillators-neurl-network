import numpy as np

from data_loader import DataLoader
from utilities.activation_functions import FUNC, DERIVATIVE
from utilities.activation_functions import sigmoid
from utilities.constants import *


# Depth: 2
# Width:
#   First layer: 1024
#   Second layer: 512
#   Third layer: 128
#   Output layer

# Runs on epochs, CNN Neural network
class CNNNeuralNetwork:

    def __init__(self, activation_func, learning_rate):
        self.activation_func = activation_func
        self.learning_rate = learning_rate

        np.random.seed(RANDOM_SEED)
        # Initialize synapse layer with values in the range [-1, 1] with mean zero
        self.synapse_0 = 2 * np.random.random((WIDTH_LAYER_1, WIDTH_LAYER_2)) - 1
        self.synapse_1 = 2 * np.random.random((WIDTH_LAYER_2, WIDTH_LAYER_3)) - 1
        self.synapse_2 = 2 * np.random.random((WIDTH_LAYER_3, 1)) - 1

    def run_epoch(self, data_loader: DataLoader):
        activate, derivative = self.activate, self.derivative

        epoch_batch_loss = 0
        for _filename, batch_samples, batch_expected in data_loader:
            # Feed forward
            layer_1_output = activate(np.dot(batch_samples, self.synapse_0))
            layer_2_output = activate(np.dot(layer_1_output, self.synapse_1))
            batch_prediction = sigmoid(np.dot(layer_2_output, self.synapse_2))  # logistic sigmoid

            # errors
            batch_output_error = batch_expected - batch_prediction

            # collect stats
            epoch_batch_loss += np.sum(np.power(batch_output_error, 2))

            # Backpropagation
            batch_output_delta = batch_output_error * derivative(batch_prediction)
            layer_2_error = batch_output_delta.dot(self.synapse_2.T)
            layer_2_delta = layer_2_error * derivative(layer_2_output)
            layer_1_error = layer_2_delta.dot(self.synapse_1.T)
            layer_1_delta = layer_1_error * derivative(layer_1_output)
            # update weights (synapses)
            self.synapse_2 += self.learning_rate * layer_2_output.T.dot(batch_output_delta)
            self.synapse_1 += self.learning_rate * layer_1_output.T.dot(layer_2_delta)
            self.synapse_0 += self.learning_rate * batch_prediction.T.dot(layer_1_delta)

        # for stats
        return epoch_batch_loss

    def train_return_acc_and_loss(self, epochs, print_every=25):
        data_loader = DataLoader()
        acc = []
        loss = []
        test_acc = []
        test_loss = []
        weights = []

        for e in range(epochs):
            weights.append((np.copy(self.synapse_0), np.copy(self.synapse_1), np.copy(self.synapse_2)))
            epoch_loss = self.run_epoch(data_loader)
            half_avg_epoch_loss = epoch_loss / (2 * len(data_loader))
            avg_epoch_acc = (1 - 2 * half_avg_epoch_loss) * 100

            epoch_test_acc, epoch_test_loss = self.predict_samples_return_acc_and_loss(
                data_loader.get_samples_with_expected_result(data_loader.test_set))
            test_loss.append(epoch_test_loss)
            test_acc.append(epoch_test_acc)

            if e % print_every == 0:
                print(f"**[{e + 1} Epoch]**")
                print(f"avg_acc = {avg_epoch_acc}")
                print(f"avg_loss = {half_avg_epoch_loss}")
                print(f"test_acc = {epoch_test_acc}")
                print(f"test_loss = {epoch_test_loss}")
            acc.append(avg_epoch_acc)
            loss.append(half_avg_epoch_loss)

        return acc, loss, test_acc, test_loss, weights

    def predict_samples_return_acc_and_loss(self, files_samples_and_expected, synapse0=None, synapse1=None,
                                            synapse2=None):
        expected, results = self.predict(files_samples_and_expected, synapse0, synapse1, synapse2)

        loss = np.sum(np.power(expected - results, 2)) / (2 * len(results))
        acc = (1 - 2 * loss) * 100
        return acc, loss

    def predict(self, files_samples_and_expected, synapse0=None, synapse1=None,
                synapse2=None):
        synapse0 = self.synapse_0 if synapse0 is None else synapse0
        synapse1 = self.synapse_1 if synapse1 is None else synapse1
        synapse2 = self.synapse_2 if synapse2 is None else synapse2

        _, samples, expected = files_samples_and_expected
        layer_1_output = self.activate(np.dot(samples, synapse0))
        layer_2_output = self.activate(np.dot(layer_1_output, synapse1))
        batch_output = np.dot(layer_2_output, synapse2)
        batch_prediction = sigmoid(batch_output)

        return expected, batch_prediction

    def activate(self, x):
        return self.activation_func[FUNC](x)

    def derivative(self, x):
        return self.activation_func[DERIVATIVE](x)
