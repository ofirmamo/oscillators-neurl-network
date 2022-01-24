import os
from pathlib import Path

import numpy as np

import rle_decoder as decoder
from utilities.constants import DATA_SOURCES_PATH, OSCILLATORS_PATH, NON_OSCILLATORS_PATH
from utilities.utilities import trim_zeros

SOURCES_FOLDER = os.path.join("..", "..", DATA_SOURCES_PATH, "rle")
OSCILLATORS_FOLDER = os.path.join("..", OSCILLATORS_PATH)
NON_OSCILLATORS_FOLDER = os.path.join("..", NON_OSCILLATORS_PATH)

relevant_file_names = []
for file_name in os.listdir(SOURCES_FOLDER):
    with open(os.path.join(SOURCES_FOLDER, file_name)) as file:
        raw_content = file.read()
        header = decoder.decode_header(raw_content)  # get size
        if header.height <= 32 and header.width <= 32:  # already fit
            relevant_file_names.append(file_name)
            continue
        matrix = decoder.decode(raw_content)

        failed = 0
        try:
            matrix_trimmed = np.asmatrix(trim_zeros(matrix))  # trim and try to fit
            shape = matrix_trimmed.shape
            if shape[0] <= 32 and shape[1] <= 32:
                relevant_file_names.append(file_name)
        except:
            failed += 1
            pass

# filter oscillators
oscillator_files = []
for file_name in relevant_file_names:
    with open(os.path.join(SOURCES_FOLDER, file_name)) as file:
        lines = file.readlines()
        if any("oscillator" in line.lower() for line in lines):
            oscillator_files.append(file_name)

# put in folders
Path(OSCILLATORS_FOLDER).mkdir(parents=True, exist_ok=True)
Path(NON_OSCILLATORS_FOLDER).mkdir(parents=True, exist_ok=True)
for file_name in relevant_file_names:
    if file_name in oscillator_files:
        os.rename(os.path.join(SOURCES_FOLDER, file_name), os.path.join(OSCILLATORS_FOLDER, file_name))
    else:
        os.rename(os.path.join(SOURCES_FOLDER, file_name), os.path.join(NON_OSCILLATORS_FOLDER, file_name))
