import os

# region board
BOARD_SIZE = 32
BOARD_SHAPE = (BOARD_SIZE, BOARD_SIZE)
# endregion

# region neural network
BATCH_SIZE = 132
AUGMENTATIONS = True
TRAIN_MODE = "Train"
TEST_MODE = "Test"
RANDOM_SEED = 1
WIDTH_LAYER_1 = BOARD_SIZE * BOARD_SIZE
WIDTH_LAYER_2 = 512
WIDTH_LAYER_3 = 128
# endregion

# region Thresholds
OSCILLATOR_PREDICTION_THRESHOLD = 0.5
# endregion

# region paths
DATA_SOURCES_PATH = "data-sources"
OSCILLATORS = "oscillators"
NON_OSCILLATORS = "non_oscillators"
OSCILLATORS_PATH = os.path.join(DATA_SOURCES_PATH, OSCILLATORS)
NON_OSCILLATORS_PATH = os.path.join(DATA_SOURCES_PATH, NON_OSCILLATORS)
# endregion