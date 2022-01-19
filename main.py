import matplotlib.pyplot as plt
import numpy as np

import utilities.activation_functions as actf
from data_loader import DataLoader
from neural_network import CNNNeuralNetwork
from utilities.constants import OSCILLATOR_PREDICTION_THRESHOLD


def plot_acc_and_loss(train_accuracy, test_accuracy, epoch_of_best_test_acc):
    plt.plot(train_accuracy, label="Train Accuracy")
    plt.plot(test_accuracy, label="Test Accuracy")
    plt.ylabel(f'Accuracy')
    plt.xlabel("Epochs:")
    plt.legend()
    plt.axvline(x=epoch_of_best_test_acc, color="gray")
    plt.show()


# region training
neural_network = CNNNeuralNetwork(activation_func=actf.FUNCTIONS[actf.SIGMOID], learning_rate=0.0001)
data_loader = DataLoader()

(train_acc, train_loss,
 test_acc, test_loss,
 weights) = neural_network.train_return_acc_and_loss(epochs=12000, print_every=100)
# endregion


# region best accuracy.
print(f"num of epochs until overfit: {(epoch_of_max_test_acc := test_acc.index(max(test_acc)))}")
synapse0, synapse1, synapse2 = weights[epoch_of_max_test_acc]
# endregion

# region performance plots
plot_acc_and_loss(train_acc, test_acc, epoch_of_max_test_acc)
# endregion


# region failures
train_failures = []

files, samples, expected = data_loader.get_samples_with_expected_result(data_loader.train_set)
for file, sample, expected in zip(files, samples, expected):
    sample = np.reshape(sample, (1, 1024))
    prediction = neural_network.predict_samples((sample, expected), synapse0, synapse1, synapse2)
    if prediction[0][0] >= OSCILLATOR_PREDICTION_THRESHOLD and expected[0] == 0.0:
        train_failures.append((file, expected[0]))
    if prediction[0][0] <= 1 - OSCILLATOR_PREDICTION_THRESHOLD and expected[0] == 1.0:
        train_failures.append((file, expected[0]))

test_failures = []
files, samples, expected = data_loader.get_samples_with_expected_result(data_loader.test_set)
for file, sample, expected in zip(files, samples, expected):
    sample = np.reshape(sample, (1, 1024))
    prediction = neural_network.predict_samples((sample, expected), synapse0, synapse1)
    if prediction[0][0] >= OSCILLATOR_PREDICTION_THRESHOLD and expected[0] == 0.0:
        test_failures.append((file, expected[0]))
    if prediction[0][0] <= 1 - OSCILLATOR_PREDICTION_THRESHOLD and expected[0] == 1.0:
        test_failures.append((file, expected[0]))

print(f"""
    ** Train score **
    Percentage of failures: {(len(train_failures) / len(data_loader.train_set)) * 100}
    Number of failures: {len(train_failures)}
    Number of tests: {len(data_loader.train_set)}
    List: {train_failures}
""")

print(f"""
    ** Test score **
    Percentage of failures: {(len(test_failures) / len(data_loader.test_set)) * 100}
    Number of failures: {len(test_failures)}
    Number of tests: {len(data_loader.test_set)}
    List: {test_failures}
""")
# endregion failures