# Checks the shape of the concatenated data saved in the derivatives folder

import mne
import numpy as np

# Set parameters for the concatenated file path
stim = 'Jobs1'
run = 'run-01'
freq = '70-150Hz'
pro = 'preprocessed'

# Set the path to the saved concatenated file
concatenated_file = f'/derivatives/2-concatenated/{pro}/{freq}/{stim}_{run}_concatenated_raw.fif'

# Read the concatenated raw data from the saved file
concatenated_raw = mne.io.read_raw_fif(concatenated_file)

# Get the shape of the concatenated raw data
num_channels, num_time_points = concatenated_raw.get_data().shape

# Print the shape of the concatenated raw data
print(f"Shape of the concatenated raw data: {num_channels} channels, {num_time_points} time points")

# Print the channel names
print("Channel names in the concatenated raw data:")
print(concatenated_raw.ch_names)
print(np.shape(concatenated_raw.get_data()))