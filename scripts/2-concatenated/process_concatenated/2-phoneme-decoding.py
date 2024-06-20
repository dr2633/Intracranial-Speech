import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
import mne
from mne.decoding import ReceptiveField, TimeDelayingRidge

# Set parameter for processing type
pro = 'preprocessed'
freq = '70-150Hz'

# Set all stimuli and runs for analysis
stimuli = ['Jobs1', 'Jobs2', 'Jobs3', 'AttFast', 'AttSlow', 'BecFast', 'BecSlow', 'CampFast', 'CampSlow']

runs = ['run-01', 'run-02']

# Set base path depending on who is running the code
user_paths = ['/Users/derekrosenzweig/Documents/GitHub/iEEG-Speech',
              '/Users/lauragwilliams/Documents/projects/iEEG-Speech']

for user_path in user_paths:
    if os.path.exists(user_path):
        base_path = user_path


# Set figure path
fig_path = os.path.join(base_path, 'vis', 'mega','phoneme-decoding')
os.makedirs(fig_path, exist_ok=True)

# Loop through each unique stim and run combination
for stim in stimuli:
    for run in runs:
        # Construct file path for concatenated FIF file
        fif_path = os.path.join(base_path, 'derivatives', 'mega', pro, freq, f'{stim}_{run}_concatenated_raw.fif')

        # Check if file exists
        if not os.path.exists(fif_path):
            print(f"File does not exist: {fif_path}, skipping...")
            continue

        print(f"Processing file: {fif_path}")


