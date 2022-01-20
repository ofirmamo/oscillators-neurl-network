import numpy as np

FUNC = "f"
DERIVATIVE = "df"

BINARY_STEP = "binary_step"
SIGMOID = "sigmoid"
HYPER_TANH = "tanh"


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


FUNCTIONS = {
    BINARY_STEP:
        {
            FUNC: lambda x: np.where(x < 0, 0, 1),
            DERIVATIVE: lambda x: np.where(x != 0, 0, np.NAN)
        },
    SIGMOID:
        {
            FUNC: sigmoid,
            DERIVATIVE: lambda x: sigmoid(x) * (1 - sigmoid(x))
        },
    HYPER_TANH:
        {
            FUNC: np.tanh,
            DERIVATIVE: lambda x: 1 - np.power(np.tanh(x), 2)
        }
}
