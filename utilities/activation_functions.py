import numpy as np

FUNC = "f"
DERIVATIVE = "df"

BINARY_STEP = "binary_step"
SIGMOID = "sigmoid"
HYPER_TANH = "tanh"
RELU = "relu"
SOFTPLUS = "softplus"
LEAKY_RELU = "leaky_relu"


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _derivative_binary_step(x):
    if x == 0:
        raise ValueError("x = 0")
    return 0


def relu_derivative(x):
    if x >= 0:
        return 1
    return 0


FUNCTIONS = {
    BINARY_STEP:
        {
            FUNC: lambda x: np.where(x < 0, 0, 1),
            DERIVATIVE: lambda x: _derivative_binary_step
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
        },
    RELU:
        {
            FUNC: lambda x: (relu_ed := np.maximum(x, 0)) / np.sum(relu_ed),
            DERIVATIVE: lambda x: (x > 0) * 1
        },
    SOFTPLUS:
        {
            FUNC: lambda x: np.log(1 + np.exp(x)),
            DERIVATIVE: lambda x: 1 / (1 + np.exp(-x))
        },
    LEAKY_RELU:
        {
            FUNC: lambda x: np.where(x > 0, x, x * 0.01),
            DERIVATIVE: lambda x: np.where(x < 0, 0.01, x)
        }
}
