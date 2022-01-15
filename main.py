import matplotlib.pyplot as plt
import numpy as np

import utilities.activation_functions as actf
from data_loader import DataLoader
from neural_network import CNNNeuralNetwork
from utilities.constants import TEST_MODE, TRAIN_MODE, OSCILLATOR_PREDICTION_THRESHOLD


def plot_acc_and_loss(mode, acc, loss):
    plt.plot(acc)
    plt.title(mode)
    plt.ylabel(f'Accuracy')
    plt.xlabel("Epochs:")
    plt.show()

    # plotting Loss
    plt.plot(loss)
    plt.ylabel('Loss')
    plt.xlabel("Epochs:")
    plt.show()


# region training
neural_network = CNNNeuralNetwork(activation_func=actf.FUNCTIONS[actf.SIGMOID], learning_rate=0.0001)
data_loader = DataLoader()

train_acc, train_loss, test_acc, test_loss, weights = neural_network.train_return_acc_and_loss(epochs=10000,
                                                                                               print_every=100)
# endregion

# region performance plots
plot_acc_and_loss(TRAIN_MODE, train_acc, train_loss)
plot_acc_and_loss(TEST_MODE, test_acc, test_loss)
# endregion

# region choose the best accuracy.
print(f"num of epochs until overfit: {(idx_of_max_test_acc := test_acc.index(max(test_acc)))}")
synapse0, synapse1 = weights[idx_of_max_test_acc]

# endregion

train_failures = []

files, samples, expected = data_loader.get_samples_with_expected_result(data_loader.train_set)
for file, sample, expected in zip(files, samples, expected):
    sample = np.reshape(sample, (1, 1024))
    prediction = neural_network.predict_samples((sample, expected), synapse0, synapse1)
    if prediction[0][0] >= OSCILLATOR_PREDICTION_THRESHOLD and expected[0] == 0.0:
        train_failures.append((file, expected[0]))
    if prediction[0][0] < 100 - OSCILLATOR_PREDICTION_THRESHOLD and expected[0] == 1.0:
        train_failures.append((file, expected[0]))

test_failures = []
files, samples, expected = data_loader.get_samples_with_expected_result(data_loader.test_set)
for file, sample, expected in zip(files, samples, expected):
    sample = np.reshape(sample, (1, 1024))
    prediction = neural_network.predict_samples((sample, expected), synapse0, synapse1)
    if prediction[0][0] >= OSCILLATOR_PREDICTION_THRESHOLD and expected[0] == 0.0:
        test_failures.append((file, expected[0]))
    if prediction[0][0] < 100 - OSCILLATOR_PREDICTION_THRESHOLD and expected[0] == 1.0:
        test_failures.append((file, expected[0]))

print(f"""
    ** Train score **
    Percentage of failures: {(len(train_failures) / len(data_loader.train_set)) * 100}
    Number of failures: {len(train_failures)}
    List: {train_failures}
""")

print(f"""
    ** Test score **
    Percentage of failures: {(len(test_failures) / len(data_loader.test_set)) * 100}
    Number of failures: {len(test_failures)}
    List: {test_failures}
""")