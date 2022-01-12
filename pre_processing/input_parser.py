import json
import os
from pathlib import Path
import numpy as np
import lifeforms_gen.rle_decoder as decoder





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
