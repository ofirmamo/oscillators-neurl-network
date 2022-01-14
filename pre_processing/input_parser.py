import json
import os
from pathlib import Path
import numpy as np
import lifeforms_gen.rle_decoder as decoder


def trim_zeros(arr):
    """Returns a trimmed view of an n-D array excluding any outer
    regions which contain only zeros.
    """
    slices = tuple(slice(idx.min(), idx.max() + 1) for idx in np.nonzero(arr))
    return arr[slices]

PNGS_FILE = "pngs.json"
SOURCES_FOLDER = os.path.join("..","data-sources", "rle")
OSCILLATORS_FOLDER = os.path.join("..","data-sources", "oscillators")
NON_OSCILLATORS_FOLDER = os.path.join("..","data-sources", "non_oscillators")

relevant_file_names = []
for file_name in os.listdir(SOURCES_FOLDER):
    with open(os.path.join(SOURCES_FOLDER, file_name)) as file:
        raw_content = file.read()
        header = decoder.decode_header(raw_content)
        if header.height <= 32 and header.width <= 32:
            relevant_file_names.append(file_name)
            continue
        matrix = decoder.decode(raw_content)

        failed = 0
        try:
            matrix_trimmed = np.asmatrix(trim_zeros(matrix))
            shape = matrix_trimmed.shape
            if shape[0] <= 32 and shape[1] <= 32:
                relevant_file_names.append(file_name)
        except:
            failed += 1
            pass

print(len(relevant_file_names))

with open(PNGS_FILE) as json_file:
    data = json.load(json_file)
    oscillators_urls = data["Oscillators"]

oscillator_from_urls_list = [url.rsplit('/', 1)[-1].replace(".png", ".rle").lower() for url in oscillators_urls]

oscillator_by_comment = []
for file_name in relevant_file_names:
    with open(os.path.join(SOURCES_FOLDER, file_name)) as file:
        lines = file.readlines()
        if any("oscillator" in line.lower() for line in lines):
            oscillator_by_comment.append(file_name)

all_oscillator_files = (set(relevant_file_names).intersection(set(oscillator_from_urls_list))).union(
    set(oscillator_by_comment))



print(len(set(relevant_file_names).intersection(set(oscillator_by_comment))))
print(len(oscillator_by_comment))
print(set(oscillator_by_comment).difference(set(relevant_file_names)))
print(oscillator_by_comment)
print(set(relevant_file_names).difference(set(oscillator_by_comment)))

Path(OSCILLATORS_FOLDER).mkdir(parents=True, exist_ok=True)
Path(NON_OSCILLATORS_FOLDER).mkdir(parents=True, exist_ok=True)
for file_name in relevant_file_names:
    if file_name in all_oscillator_files:
        os.rename(os.path.join(SOURCES_FOLDER, file_name), os.path.join(OSCILLATORS_FOLDER, file_name))
    else:
        os.rename(os.path.join(SOURCES_FOLDER, file_name), os.path.join(NON_OSCILLATORS_FOLDER, file_name))
