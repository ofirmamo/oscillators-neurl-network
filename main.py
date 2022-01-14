import utilities.activation_functions as actf
from data_loader import DataLoader
from neural_network import CNNNeuralNetwork
from utilities.error_functions import FUNCTIONS, L2_ERROR

neural_network = CNNNeuralNetwork(activation_func=actf.FUNCTIONS[actf.SIGMOID], learning_rate=0.001)
data_loader = DataLoader()

samples, expected = data_loader.get_samples_with_expected_result()

for i in range(100001):
    neural_network.run_epoch(data_loader)
    if i % 25 == 0:
        print(f"[round {i+1}] error: {FUNCTIONS[L2_ERROR](expected, neural_network.predict_samples(samples))}")