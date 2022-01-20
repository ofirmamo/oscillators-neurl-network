from lifeforms_gen import rle_decoder as decoder
from utilities.utilities import *
from data_loader import DataLoader

data_loader = DataLoader()

with open("data-sources/non_oscillators/mwss.rle") as f:
    raw_content = f.read()
    tested_raw_lf = decoder.decode(raw_content)

min_dist = np.linalg.norm(tested_raw_lf - data_loader.oscillators_learning_samples_with_expected[1])
file = data_loader.oscillators_learning_samples_with_expected[0]
for file_path, raw_lf, _ in data_loader.oscillators_learning_samples_with_expected[1:]:
    dist = np.linalg.norm(tested_raw_lf - raw_lf)
    if dist < min_dist:
        file = file_path
        min_dist = dist

print(f'Minimum distance for an oscillator is: {min_dist}, rle file: {file}')
