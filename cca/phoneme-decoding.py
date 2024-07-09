import os
import pandas as pd
import scipy.stats as stats
import numpy as np
import mne
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Parameters
ANALYSIS_TYPES = ['F0', 'ENTROPY', 'INTENSITY', 'EMBEDDING_1', 'EMBEDDING_2', 'EMBEDDING_3', 'EMBEDDING_4', 'EMBEDDING_5']
freq = '70-150Hz'
stim = 'Jobs1'
run = 'run-01'

# Paths
base_path = '/Users/derekrosenzweig/PycharmProjects/CCA-reduce'
weights_path = os.path.join(base_path, 'cca-weights', 'weights')
master_electrodes_path = os.path.join(base_path, 'electrodes', 'master-electrodes.tsv')
file_path = os.path.join(base_path, 'derivatives', 'concatenated', '70-150Hz', 'preprocessed', f'{stim}_{run}_concatenated_raw.fif')
phoneme_path = os.path.join(base_path, 'annotations', 'phonemes', 'tsv', f'{stim}-phonemes.tsv')
fig_path = os.path.join(base_path, 'figures', 'phoneme-decoding')
os.makedirs(fig_path, exist_ok=True)

# Load master electrodes file
master_electrodes = pd.read_csv(master_electrodes_path, delimiter='\t')

# Load iEEG data
raw = mne.io.read_raw_fif(file_path, preload=True)
ieeg_data = raw.get_data()

# Load phoneme annotations
phoneme_data = pd.read_csv(phoneme_path, delimiter='\t')

# Set the channel names in raw iEEG data to sub_name values from master electrodes
channel_mapping = {raw.ch_names[i]: master_electrodes['sub_name'].iloc[i] for i in range(len(raw.ch_names))}
raw.rename_channels(channel_mapping)

# Function to get torch tensors for significant electrodes
def get_significant_tensors(analysis_type):
    weights_file_path = os.path.join(weights_path, f'70-150Hz_CCA_weights_{analysis_type}.csv')

    # Load weights data
    weights_data = pd.read_csv(weights_file_path)

    # Get significant electrodes
    z_scores = stats.zscore(weights_data['weight'])
    weights_data['z_score'] = z_scores
    significant_data = weights_data[np.abs(weights_data['z_score']) > 2]

    # Filter significant electrodes to match sub_name in master-electrodes
    matched_significant_data = significant_data[significant_data['sub_name'].isin(master_electrodes['sub_name'])]

    # Print matched significant electrode sub_names from weights CSV
    print(f"Matched significant electrodes in {analysis_type} weights CSV:")
    for sub_name in matched_significant_data['sub_name']:
        print(sub_name)

    # Print index of these rows in the master-electrodes file
    print(f"\nIndices of matched significant electrodes in master-electrodes for {analysis_type}:")
    indices = []
    for sub_name in matched_significant_data['sub_name']:
        idx = master_electrodes.index[master_electrodes['sub_name'] == sub_name].tolist()
        indices.extend(idx)
        print(f"{sub_name}: {idx}")

    # Get the data for the matched electrodes
    matched_indices = [raw.ch_names.index(name) for name in matched_significant_data['sub_name'] if name in raw.ch_names]
    filtered_ieeg_data = ieeg_data[matched_indices, :]

    # Convert to PyTorch tensor
    ieeg_tensor = torch.tensor(filtered_ieeg_data, dtype=torch.float32)

    return ieeg_tensor

# Create a dictionary to store tensors for each analysis type
tensors_dict = {}
for analysis_type in ANALYSIS_TYPES:
    tensor = get_significant_tensors(analysis_type)
    tensors_dict[analysis_type] = tensor
    print(f"Shape of tensor for {analysis_type}: {tensor.shape}")

print("Finished creating torch tensors for each feature set.")

# Sklearn wrapper for Logistic Regression
clf = make_pipeline(StandardScaler(), LogisticRegression(solver='liblinear'))
cv = KFold(5, shuffle=True)

# Function to run analysis for a specific type
def run_analysis(analysis_type):
    # Get the tensor for the current analysis type
    ieeg_tensor = tensors_dict[analysis_type]

    # Convert tensor to numpy array for sklearn
    ieeg_data_np = ieeg_tensor.numpy()

    # Create info object for the subset
    ch_names = [raw.ch_names[i] for i in range(ieeg_data_np.shape[0])]
    info_subset = mne.create_info(ch_names=ch_names, sfreq=raw.info['sfreq'], ch_types=['seeg'] * len(ch_names))

    # Create RawArray object for the subset
    raw_subset = mne.io.RawArray(ieeg_data_np, info_subset)

    # Convert start times to samples assuming a sampling frequency of 1000 Hz
    phoneme_onsets = (phoneme_data['Start'].values * raw.info['sfreq']).astype(int)

    # Create events for phoneme epochs
    phoneme_events = np.column_stack((phoneme_onsets, np.zeros_like(phoneme_onsets), np.ones_like(phoneme_onsets)))

    # Ensure the number of events matches the number of metadata rows
    min_len = min(len(phoneme_events), len(phoneme_data))
    phoneme_events = phoneme_events[:min_len]
    phoneme_data_trimmed = phoneme_data[:min_len]

    # Create epochs
    epochs = mne.Epochs(raw_subset, phoneme_events, event_id=None, tmin=-0.2, tmax=0.6, preload=True, baseline=None,
                        event_repeated='drop')

    # Set metadata
    epochs.metadata = phoneme_data_trimmed

    # Resample to 100 Hz
    epochs.resample(100)

    # Specify the desired values for decoding
    desired_values = {
        'phonation': 'v',
        'manner': 'f',
        'place': 'm',
        'roundness': 'r',
        'frontback': 'f'
    }

    # Colors for each phoneme feature
    colors = {
        'phonation': 'darkgreen',
        'manner': 'green',
        'place': 'lightgreen',
        'roundness': 'lime',
        'frontback': 'palegreen'
    }

    # Plotting
    y_min = 0.45
    y_max = 1

    plt.figure(figsize=(12, 6))

    for feat, color in colors.items():
        desired_value = desired_values[feat]  # Get the desired value for the current feature

        y = (epochs.metadata[feat] == desired_value).astype(int)  # Extract the labels for the current feature

        # Preallocate accuracy scores array
        accuracy_scores = np.empty(epochs.get_data(copy=True).shape[-1])

        # Extract data and compute ROC-AUC scores across time
        for tt in range(accuracy_scores.shape[0]):
            X_ = epochs.get_data(copy=True)[:, :, tt]
            scores = cross_val_score(clf, X_, y, scoring='roc_auc', cv=cv, n_jobs=-1)
            accuracy_scores[tt] = scores.mean()

        plt.plot(epochs.times, accuracy_scores, label=feat.capitalize(), color=color)

    plt.axhline(y=0.5, color='grey', linestyle='--')  # Chance level decoding
    plt.title(f'{analysis_type} - Distribution of Anatomical Sites', fontsize=18, fontweight='bold')
    plt.xlabel('Time (ms) relative to phoneme onset', fontsize=14)
    plt.ylabel('ROC-AUC', fontsize=14)
    plt.ylim(y_min, y_max)  # Set y-axis limits
    plt.legend()
    plt.tight_layout()

    # Save figure
    plt.savefig(os.path.join(fig_path, f'{analysis_type}_{freq}_{stim}_{run}_logistic.jpg'), dpi=300)
    plt.show()

    print(f"Decoding analysis completed for {analysis_type}.")

# Loop through each analysis type and run the analysis
for analysis_type in ANALYSIS_TYPES:
    run_analysis(analysis_type)

print("All analyses completed.")
