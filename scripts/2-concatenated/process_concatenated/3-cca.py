import mne
import numpy as np
import pandas as pd
import scipy.signal as signal
from sklearn.cross_decomposition import CCA
import os

#Params
stim = 'AttFast'
run = 'run-01'

# Load data for audio features (Mel coefficients)
audio = f'/Users/derekrosenzweig/Documents/GitHub/iEEG-Speech/annotations/acoustics/tsv/{stim}-acoustics.tsv'

aud = pd.read_csv(audio, delimiter='\t')

# Load data for iEEG (channels across timepoints)
ieeg = f'/Users/derekrosenzweig/Documents/GitHub/iEEG-Speech/derivatives/mega/preprocessed/70-150Hz/{stim}_{run}_concatenated_raw.fif'

raw = mne.io.read_raw_fif(ieeg, preload=True)
ieeg_data = raw.get_data()

# Downsample the iEEG data to 100 Hz (currently sampled at 1000 Hz)
ieeg_data_downsampled = signal.resample(ieeg_data, num=ieeg_data.shape[1] // 10, axis=1)

# Extract the 'F0 (Hz)' feature from the audio DataFrame and replace NaN values with 0
f0 = aud['F0 (Hz)'].fillna(0).values

# Transpose the data matrices to have shape (n_samples, n_features)
ieeg_data_transposed = ieeg_data_downsampled.T
f0_transposed = f0.reshape(-1, 1)

# Create a CCA object and fit the data
n_components = 1  # Using only one component for the simple case
cca = CCA(n_components=n_components)
cca.fit(ieeg_data_transposed, f0_transposed)

# Transform the data using the fitted CCA model
X_transformed, Y_transformed = cca.transform(ieeg_data_transposed, f0_transposed)

# Compute the canonical correlations
canonical_correlations = np.corrcoef(X_transformed.T, Y_transformed.T)[0, 1]

print("Canonical Correlations:", canonical_correlations)

# Get the canonical weights for iEEG data
ieeg_weights = cca.x_loadings_

# Print the canonical weights for each electrode
for i, electrode in enumerate(raw.ch_names):
    print(f"Electrode {electrode}: {ieeg_weights[i, 0]}")


# Rank the electrodes based on the absolute values of their canonical weights
electrode_rankings = np.argsort(np.abs(ieeg_weights[:, 0]))[::-1]

# Print the top N electrodes with their weights
N = 10
print(f"Top {N} electrodes maximally correlated with F0:")
for i in range(N):
    electrode_index = electrode_rankings[i]
    electrode_name = raw.ch_names[electrode_index]
    electrode_weight = ieeg_weights[electrode_index, 0]
    print(f"{i+1}. {electrode_name}: {electrode_weight}")

# Directory to save the CSV file
csv_directory = "/Users/derekrosenzweig/Documents/GitHub/iEEG-Speech/mega_scripts/cca"

# Create the directory if it doesn't exist
os.makedirs(csv_directory, exist_ok=True)
save_file = f'{stim}_{run}'

# Save the ordered list of electrodes and their weights to a CSV file
csv_filename = os.path.join(csv_directory, f"{save_file}_electrode_rankings.csv")
with open(csv_filename, "w") as file:
    file.write("Electrode,Weight\n")
    for electrode_index in electrode_rankings:
        electrode_name = raw.ch_names[electrode_index]
        electrode_weight = ieeg_weights[electrode_index, 0]
        file.write(f"{electrode_name},{electrode_weight}\n")

print(f"Ordered list of electrodes and weights saved to {csv_filename}")


