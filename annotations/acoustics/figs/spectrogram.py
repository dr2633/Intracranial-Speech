import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Set parameters
stim = 'CampSlow'
TSV_FILE_PATH = f'/Users/derekrosenzweig/Documents/GitHub/ieeg/iEEG_BIDS/sEEG_BIDS/annotations/acoustics/tsv/{stim}-acoustics.tsv'

# Load the TSV file
data = np.loadtxt(TSV_FILE_PATH, delimiter='\t', skiprows=1)  # Skip header row

# Extract time values and Mel energy coefficients
time_values_sec = data[:, 0]
mel_energy_coeffs = data[:, 16:]

# Reshape Mel energy coefficients to have the correct shape
num_mel_coeffs = mel_energy_coeffs.shape[1]
num_timepoints = len(time_values_sec)
mel_energy_coeffs_reshaped = mel_energy_coeffs.T.reshape(num_mel_coeffs, num_timepoints)

# Adjust log power values
mel_array = np.log(mel_energy_coeffs_reshaped + 1e-9)  # Add a small epsilon to avoid taking the log of zero
mel_array = (mel_array - mel_array.min()) / (mel_array.max() - mel_array.min())  # Scale to range [0, 1]
mel_array = mel_array * (15 - (-30)) + (-30)  # Scale to range [-30, 15]

# Adjust arrays to display the first 10 seconds
end_index = int(10 / (time_values_sec[1] - time_values_sec[0]))  # Convert 10 seconds to index
time_values_sec = time_values_sec[:end_index]
mel_array = mel_array[:, :end_index]

# Plot the Mel spectrogram
plt.figure(figsize=(10, 4))
plt.imshow(mel_array, aspect='auto', origin='lower', cmap='viridis',
           extent=[0, time_values_sec[-1], 0, mel_array.shape[0]])
plt.xlabel('Time (seconds)')
plt.ylabel('Mel Filterbank Index')
plt.title('Mel Spectrogram (First 10 Seconds)')
plt.colorbar(label='Log Power (dB)')

# Save the figure
output_dir = '/Users/derekrosenzweig/Documents/GitHub/ieeg/iEEG_BIDS/sEEG_BIDS/annotations/acoustics/figs'
output_filename = f'{stim}.jpg'
output_path = os.path.join(output_dir, output_filename)
plt.savefig(output_path)

# Show the plot
plt.show()
