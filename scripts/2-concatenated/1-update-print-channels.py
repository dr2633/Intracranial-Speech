# Script prints channel names for each trial run

import mne
import numpy as np
import pandas as pd


# List all sessions, stimuli, and runs to loop through
stimuli = ['Jobs1', 'Jobs2', 'Jobs3', 'AttFast', 'AttSlow', 'BecFast', 'BecSlow', 'CampFast', 'CampSlow']

runs = ['run-01', 'run-02']


# Set parameters for path
stim = 'Jobs1'
ses = 'ses-01'
run = 'run-01'
freq = '1-40Hz'
pro = 'preprocessed'

base_path = f'/Users/derekrosenzweig/Documents/GitHub/iEEG-Speech/derivatives/individual/{freq}/preprocessed'

# Create an empty list to store file paths
file_paths = []

# Create a for loop that gets files for sub = 'sub-01' through 'sub-07'
for i in range(2, 9):  # Iterate through sub-01 to sub-07
    if i == 4 and ses == 'ses-02':
        continue
    sub = f'sub-{i:02d}'
    fif_file = f'{base_path}/{sub}_{ses}_task-listen{stim}_{run}_ieeg.fif'
    file_paths.append(fif_file)

# Loop through each file and print out channel names
for file_path in file_paths:
    raw = mne.io.read_raw_fif(file_path)
    print(f"Channel names in file '{file_path}':")
    print(raw.ch_names)

# Loop through each file and update channel names with subject name
for sub, file_path in file_paths:
    raw = mne.io.read_raw_fif(file_path)
    for idx, channel in enumerate(raw.ch_names):
        raw.rename_channels({channel: f"{sub} {channel}"})
    print(f"Channel names in file '{file_path}' with subject '{sub}' updated.")
    print(raw.ch_names)

print("Channel name updating complete.")



