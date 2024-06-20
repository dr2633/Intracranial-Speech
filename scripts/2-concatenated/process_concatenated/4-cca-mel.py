




import mne
import numpy as np
import pandas as pd
import scipy.signal as signal
from sklearn.cross_decomposition import CCA
import os
import matplotlib.pyplot as plt

#Params
stim = 'AttFast'
run = 'run-01'
freq = '70-150Hz'

# Load data for audio features (Mel coefficients)
audio = f'/Users/derekrosenzweig/Documents/GitHub/iEEG-Speech/annotations/acoustics/tsv/{stim}-acoustics.tsv'

aud = pd.read_csv(audio, delimiter='\t')

# Load data for iEEG (channels across timepoints)
ieeg = f'/Users/derekrosenzweig/Documents/GitHub/iEEG-Speech/derivatives/concatenated/preprocessed/70-150Hz/{stim}_{run}_concatenated_raw.fif'

raw = mne.io.read_raw_fif(ieeg, preload=True)
ieeg_data = raw.get_data()

# Downsample the iEEG data to 100 Hz (currently sampled at 1000 Hz)
ieeg_data_downsampled = signal.resample(ieeg_data, num=ieeg_data.shape[1] // 10, axis=1)

# Extract the Mel Coefficients
mel_cols = [col for col in aud.columns if col.startswith('Mel_')]
mel_data = aud[mel_cols].values

# Create a CCA object and fit the data
n_components = 64  # Using 64 components for the Mel Coefficients
cca = CCA(n_components=n_components)
cca.fit(ieeg_data_downsampled.T, mel_data)

# Transform the data using the fitted CCA model
X_transformed, Y_transformed = cca.transform(ieeg_data_downsampled.T, mel_data)

# Compute the canonical correlations
canonical_correlations = np.corrcoef(X_transformed.T, Y_transformed.T)

print("Canonical Correlations:")
print(canonical_correlations)

# Get the canonical weights for iEEG data
ieeg_weights = cca.x_loadings_

# Get the canonical weights for Mel Coefficients
mel_weights = cca.y_loadings_

# Print the canonical weights for each electrode
for i, electrode in enumerate(raw.ch_names):
    print(f"Electrode {electrode}: {ieeg_weights[i, :]}")

# Print the canonical weights for each Mel Coefficient
for i, mel_coef in enumerate(mel_cols):
    print(f"Mel Coefficient {mel_coef}: {mel_weights[i, :]}")


# Identify the electrodes of interest
electrode_rankings = np.argsort(np.abs(ieeg_weights[:, 0]))[::-1]
top_electrodes = [raw.ch_names[i] for i in electrode_rankings[:10]]

# Identify the most correlated Mel Coefficients
mel_rankings = np.argsort(np.abs(mel_weights[:, 0]))[::-1]
top_mel_coefficients = [mel_cols[i] for i in mel_rankings[:10]]

# Print the results
print("Top 10 electrodes of interest:")
for electrode in top_electrodes:
    print(electrode)

print("\nTop 10 most correlated Mel Coefficients:")
for mel_coef in top_mel_coefficients:
    print(mel_coef)

# Plot the canonical correlations
plt.figure(figsize=(8, 6))
plt.plot(canonical_correlations)
plt.xlabel('Canonical Component')
plt.ylabel('Canonical Correlation')
plt.title('Canonical Correlations')
plt.grid(True)
plt.show()

# Create a bar plot of top electrodes and Mel Coefficients
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Plot top electrodes
ax1.bar(top_electrodes, np.abs(ieeg_weights[electrode_rankings[:10], 0]))
ax1.set_xticklabels(top_electrodes, rotation=45, ha='right')
ax1.set_ylabel('Weight')
ax1.set_title('Top 10 Electrodes')

# Plot top Mel Coefficients
ax2.bar(top_mel_coefficients, np.abs(mel_weights[mel_rankings[:10], 0]))
ax2.set_xticklabels(top_mel_coefficients, rotation=45, ha='right')
ax2.set_ylabel('Weight')
ax2.set_title('Top 10 Mel Coefficients')

# Adjust the layout and display the plot
plt.tight_layout()
plt.show()

# Create a heatmap of the top 100 electrode weights
top_n = 100
top_electrode_indices = electrode_rankings[:top_n]
top_electrode_names = [raw.ch_names[i] for i in top_electrode_indices]
top_electrode_weights = ieeg_weights[top_electrode_indices, 0]

# Reshape the weights into a 2D array for visualization
weights_2d = top_electrode_weights.reshape(-1, 1)

# Create a figure and axis for the heatmap
fig, ax = plt.subplots(figsize=(8, 20))

# Create the heatmap
im = ax.imshow(weights_2d, cmap='coolwarm', aspect='auto')

# Set the x-tick labels and positions
ax.set_xticks([])
ax.set_xticklabels([])

# Set the y-tick labels and positions
ax.set_yticks(np.arange(top_n))
ax.set_yticklabels(top_electrode_names)

# Add a color bar
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel('Weight', rotation=-90, va="bottom")

# Set the title
ax.set_title(f'Top {top_n} Electrode Weights')

# Adjust the layout and display the plot
plt.tight_layout()
plt.show()

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