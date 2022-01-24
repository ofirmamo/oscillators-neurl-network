import matplotlib.pyplot as plt

import utilities.activation_functions as actf
from data_loader import DataLoader
from neural_network import CNNNeuralNetwork
from utilities.constants import *

data_loader = DataLoader()

TITLE = f"Sigmoid - {'with' if AUGMENTATIONS else 'no'} Augmentations"
# TITLE = f"Sigmoid - {int(BATCH_SIZE * len(data_loader))} Batch Size"


def plot_acc_and_loss(train_accuracy, test_accuracy, epoch_of_best_test_acc):
    plt.title(TITLE)
    plt.plot(train_accuracy, label="Train Loss")
    plt.plot(test_accuracy, label="Test Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.axvline(x=epoch_of_best_test_acc, color="gray")
    plt.show()


# region training
neural_network = \
    CNNNeuralNetwork(
        activation_func=actf.FUNCTIONS[actf.SIGMOID],
        learning_rate=0.0001)

(train_acc, train_loss,
 test_acc, test_loss,
 weights) = neural_network.train_return_acc_and_loss(epochs=12000, print_every=100)
# endregion


# region best accuracy.
print(f"num of epochs until overfit: {(epoch_of_max_test_acc := test_acc.index(max(test_acc)))}")
synapse0, synapse1, synapse2 = weights[epoch_of_max_test_acc]
# endregion

# region performance plots
plot_acc_and_loss(train_loss, test_loss, epoch_of_max_test_acc)
# endregion
