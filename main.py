import utilities.activation_functions as actf
from data_loader import DataLoader
from neural_network import CNNNeuralNetwork
from utilities.error_functions import FUNCTIONS, L2_ERROR
from utilities.paths import *

neural_network = CNNNeuralNetwork(activation_func=actf.FUNCTIONS[actf.SIGMOID], learning_rate=1)
data_loader_oscillators = DataLoader(lf_type=OSCILLATORS)
data_loader_non_oscillators = DataLoader(lf_type=NON_OSCILLATORS)

oscillators_samples, oscillators_expected = data_loader_oscillators.get_samples_with_expected_result()
non_oscillators_samples, non_oscillators_expected = data_loader_non_oscillators.get_samples_with_expected_result()

for i in range(10001):
    neural_network.run_epoch(data_loaders=[data_loader_oscillators])
    if i % 25 == 0:
        print(f"[round {i+1}] oscillators error: {FUNCTIONS[L2_ERROR](oscillators_expected, neural_network.predict_samples(oscillators_samples))}")
        print(f"[round {i+1}] non oscillators error: {FUNCTIONS[L2_ERROR](non_oscillators_expected, neural_network.predict_samples(non_oscillators_samples))}")
    # if i == 99:
    #     for d in data_loader_oscillators:
    #         print(f'expected: 1, confidence result: {neural_network.predict_samples(d)}')
    #     for d in data_loader_non_oscillators:
    #         print(f'expected: 0, confidence result: {neural_network.predict_samples(d)}')