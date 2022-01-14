import numpy as np

L2_ERROR = 'l2error'

FUNCTIONS = {
    L2_ERROR: lambda expected, predicted: 0.5 * (1 / expected.shape[0]) * np.sum(np.power((expected - predicted), 2))
}
