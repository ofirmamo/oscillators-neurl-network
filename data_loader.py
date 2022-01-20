import random

import rle_decoder as decoder
from utilities.paths import *
from utilities.utilities import *


class DataLoader:
    """
    1. Load + Parse Data
    2. Trim
    3. Augment Data
    4. Padding
    5. Filter Unique
    6. Flatten
    7. Generate batching (generator of batches)
    """
    INPUT_OSCILLATOR_FILES = [os.path.join(OSCILLATORS_PATH, file_name) for file_name in os.listdir(OSCILLATORS_PATH)]
    INPUT_NON_OSCILLATOR_FILES = [os.path.join(NON_OSCILLATORS_PATH, file_name) for file_name in
                                  os.listdir(NON_OSCILLATORS_PATH)]

    def __init__(self):
        self.idx = 0
        self.oscillators_learning_samples_with_expected = []
        self.non_oscillators_learning_samples_with_expected = []

        for file_path in self.INPUT_OSCILLATOR_FILES:
            self.oscillators_learning_samples_with_expected += [
                (file_path, loaded_input, np.ones((1,))) for
                loaded_input in
                self.load_inputs(file_path)
            ]

        for file_path in self.INPUT_NON_OSCILLATOR_FILES:
            self.non_oscillators_learning_samples_with_expected += [
                (file_path, loaded_input, np.zeros((1,))) for
                loaded_input in
                self.load_inputs(file_path)
            ]

        def train_test_split():
            random.seed(RANDOM_SEED)

            pick_percent = lambda working_set: set(
                random.sample(
                    list(range(len(working_set))),
                    int(len(working_set) * 0.7)))
            pick_rest = lambda origin, picked: set(range(len(origin))) - picked

            oscillators_train_set_ids = pick_percent(self.oscillators_learning_samples_with_expected)
            non_oscillators_train_set_ids = pick_percent(self.non_oscillators_learning_samples_with_expected)
            oscillators_test_set_ids = pick_rest(
                self.oscillators_learning_samples_with_expected,
                oscillators_train_set_ids)
            non_oscillators_test_set_ids = pick_rest(
                self.non_oscillators_learning_samples_with_expected,
                non_oscillators_train_set_ids)

            train_set = [
                lst[idx] for indices, lst in
                [
                    (oscillators_train_set_ids, self.oscillators_learning_samples_with_expected),
                    (non_oscillators_train_set_ids, self.non_oscillators_learning_samples_with_expected)
                ] for idx in indices
            ]
            test_set = [
                lst[idx] for indices, lst in
                [
                    (oscillators_test_set_ids, self.oscillators_learning_samples_with_expected),
                    (non_oscillators_test_set_ids, self.non_oscillators_learning_samples_with_expected)
                ] for idx in indices
            ]
            random.shuffle(train_set)
            random.shuffle(test_set)

            return train_set, test_set

        self.train_set, self.test_set = train_test_split()
        self.working_set = self.train_set
        self.batch_size = int(len(self.working_set) * BATCH_SIZE)

    def __len__(self):
        return len(self.working_set)

    def load_inputs(self, file_path):
        with open(file_path) as file:
            raw_content = file.read()
            life_form_matrix = decoder.decode(raw_content)

            trimmed_life_form_matrix = trim_zeros(life_form_matrix)
            life_form_matrix_augmentations = self.augment_matrix(trimmed_life_form_matrix)
            life_form_padded_matrix_augmentations = [
                pad_matrix(matrix) for matrix in life_form_matrix_augmentations
            ]

            filtered = []
            for augmentation in life_form_padded_matrix_augmentations:
                if any(np.array_equal(matrix, augmentation) for matrix in filtered):
                    continue
                filtered.append(augmentation)

            life_form_flattened_augmentations = [
                flatten_matrix(matrix) for matrix in filtered
            ]

            return life_form_flattened_augmentations

    def augment_matrix(self, matrix):
        cur_matrix = matrix
        if AUGMENTATIONS:
            return [cur_matrix := np.rot90(cur_matrix) for _ in range(4)]
        else:
            return [cur_matrix]

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx >= len(self.working_set):
            self.idx = 0
            raise StopIteration()

        temp = self.idx
        self.idx += self.batch_size

        # returns list of [all_samples, all_expected] as: [(sample1, sample2...), (1, 0, ...)]
        return self.get_samples_with_expected_result(self.working_set[temp:min(len(self.working_set), self.idx)])

    def get_samples_with_expected_result(self, working_set=None):
        results = list(zip(*(working_set or self.working_set)))
        results[1] = np.stack(results[1])
        results[2] = np.asarray(results[2])
        return results
