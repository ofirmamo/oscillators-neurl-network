import numpy as np

from utilities.constants import *


def pad_matrix(matrix):
    return np.pad(matrix, ((0, (BOARD_SIZE - matrix.shape[0])), (0, (BOARD_SIZE - matrix.shape[1]))))


def flatten_matrix(matrix):
    return np.hstack(matrix)


def trim_zeros(arr):
    """Returns a trimmed view of an n-D array excluding any outer
    regions which contain only zeros.
    """
    slices = tuple(slice(idx.min(), idx.max() + 1) for idx in np.nonzero(arr))
    return arr[slices]
