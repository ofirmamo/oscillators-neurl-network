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

        np.random.seed(1)
        self.synapse_0 = 2 * np.random.random((WIDTH_LAYER_1, WIDTH_LAYER_2)) - 1
        self.synapse_1 = 2 * np.random.random((WIDTH_LAYER_2, 1)) - 1

    def run_epoch(self, data_loader: DataLoader):
        for batch_samples, batch_expected in data_loader:
            # Feed forward
            layer_1_output = self.activation_func[FUNC](np.dot(batch_samples, self.synapse_0))
            batch_prediction = self.activation_func[FUNC](np.dot(layer_1_output, self.synapse_1))

            # errors
            batch_output_error = batch_expected - batch_prediction
            batch_output_delta = batch_output_error * self.activation_func[DERIVATIVE](batch_prediction)

            layer_1_error = batch_output_delta.dot(self.synapse_1.T)
            layer_1_delta = layer_1_error * self.activation_func[DERIVATIVE](layer_1_output)

            # Backpropagation
            self.synapse_1 += self.learning_rate * layer_1_output.T.dot(batch_output_delta)
            self.synapse_0 += self.learning_rate * batch_prediction.T.dot(layer_1_delta)

    def predict_samples(self, samples):
        layer_1_output = self.activation_func[FUNC](np.dot(samples, self.synapse_0))
        return self.activation_func[FUNC](np.dot(layer_1_output, self.synapse_1))
