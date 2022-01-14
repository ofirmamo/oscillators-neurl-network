import sys

import lifeforms_gen.rle_decoder as decoder
from utilities.constants import *
from utilities.paths import *
from utilities.utilities import *


# noinspection PyMethodMayBeStatic
class DataLoader:
    """
    1. Load + Parse Data
    2. Augment Data
    4. Flatten
    4. generate batching (generator of batches)
    5.
    """
    INPUT_OSCILLATOR_FILES = [os.path.join(OSCILLATORS_PATH, file_name) for file_name in os.listdir(OSCILLATORS_PATH)]
    INPUT_NON_OSCILLATOR_FILES = [os.path.join(NON_OSCILLATORS_PATH, file_name) for file_name in
                                  os.listdir(NON_OSCILLATORS_PATH)]

    def __init__(self, lf_type=OSCILLATORS):
        self.idx = 0
        self.batch_size = 1
        self.expected_value = 1 if lf_type == OSCILLATORS else 0
        self.input_list = self.INPUT_OSCILLATOR_FILES if lf_type == OSCILLATORS else self.INPUT_NON_OSCILLATOR_FILES
        self.learning_samples = []
        for file_path in self.input_list:
            self.learning_samples += self.load_inputs(file_path)

    def load_inputs(self, file_path):
        with open(file_path) as file:
            raw_content = file.read()
            lifeform_matrix = decoder.decode(raw_content)

            trimmed_lifeform_matrix = trim_zeros(lifeform_matrix)
            # lifeform_matrix_augmentations_uniq = self.augment_matrix(trimmed_lifeform_matrix)
            # lifeform_padded_matrix_augmentations = [self.pad_matrix(matrix) for matrix in
            #                                        lifeform_matrix_augmentations_uniq]
            # TODO - Replace Man
            lifeform_padded_matrix = [self.pad_matrix(matrix) for matrix in
                                      [trimmed_lifeform_matrix]]
            filtered = []
            for augmentation in lifeform_padded_matrix:
                if any(np.array_equal(matrix, augmentation) for matrix in filtered):
                    continue
                filtered.append(augmentation)

            lifeform_flattened_augmentations = [self.flatten_matrix(matrix) for matrix in
                                                lifeform_padded_matrix]

            return lifeform_flattened_augmentations

    def augment_matrix(self, matrix):
        cur_matrix = matrix
        return [cur_matrix := np.rot90(cur_matrix) for _ in range(4)]

    def pad_matrix(self, matrix):
        return np.pad(matrix, ((0, (BOARD_SIZE - matrix.shape[0])), (0, (BOARD_SIZE - matrix.shape[1]))))

    def flatten_matrix(self, matrix):
        return np.hstack(matrix)

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx >= len(self.learning_samples):
            self.idx = 0
            raise StopIteration()

        temp = self.idx
        self.idx += self.batch_size

        return self.learning_samples[temp:min(len(self.learning_samples), self.idx + self.batch_size)]

    def get_samples_with_expected_result(self):
        return self.learning_samples, np.full(len(self.learning_samples), self.expected_value)
