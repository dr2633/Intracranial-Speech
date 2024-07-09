import os
import pandas as pd
import scipy.stats as stats
import numpy as np
import mne
import torch
from sklearn.linear_model import Ridge
from mne.decoding import SlidingEstimator, cross_val_multiscore
from sklearn.metrics import make_scorer
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

# Example function to compute Spearman correlation
def spearman_corr(y_true, y_pred):
    return spearmanr(y_true, y_pred).correlation

# Create a custom scorer for Spearman correlation
spearman_scorer = make_scorer(spearman_corr, greater_is_better=True)

# Parameters
ANALYSIS_TYPES = [ 'EMBEDDING_1']  # List of analysis types
freq = '70-150Hz'
stim = 'Jobs1'
run = 'run-01'

# Paths
base_path = '/Users/derekrosenzweig/PycharmProjects/CCA-reduce'
fig_path = os.path.join(base_path, 'figures', 'gpt-2')
os.makedirs(fig_path, exist_ok=True)
weights_path = os.path.join(base_path, 'cca-weights', 'weights')
annotation_path = os.path.join(base_path, 'annotations')
derivatives_path = os.path.join(base_path, 'derivatives')
file_path = os.path.join(base_path, 'derivatives', 'concatenated', f'{freq}', 'preprocessed', f'{stim}_{run}_concatenated_raw.fif')
master_electrodes_path = os.path.join(base_path, 'electrodes', 'master-electrodes.tsv')

# Load iEEG data
raw = mne.io.read_raw_fif(file_path, preload=True)
ieeg_data = raw.get_data()

# Read word annotations file
word_annotations_file = os.path.join(annotation_path, 'words', 'tsv', f'{stim}-words.tsv')

# Read word annotations
word_annotations = pd.read_csv(word_annotations_file, delimiter='\t')

# Load master electrodes file
master_electrodes = pd.read_csv(master_electrodes_path, delimiter='\t')

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

# Function to run analysis for a specific type
def run_analysis(analysis_type):
    ieeg_tensor = tensors_dict[analysis_type].numpy()  # Convert tensor back to numpy array

    # Create info object for the subset
    info_subset = mne.create_info(ch_names=[raw.ch_names[i] for i in range(ieeg_tensor.shape[0])], sfreq=raw.info['sfreq'], ch_types=['seeg'] * ieeg_tensor.shape[0])
    raw_subset = mne.io.RawArray(ieeg_tensor, info_subset)

    # Convert start times to samples assuming a sampling frequency of 1000 Hz
    word_onsets = (word_annotations['Start'].values * raw.info['sfreq']).astype(int)

    # Create events for word epochs
    word_events = np.column_stack((word_onsets, np.zeros_like(word_onsets), np.ones_like(word_onsets)))

    # Ensure the number of events matches the number of metadata rows
    min_len = min(len(word_events), len(word_annotations))
    word_events = word_events[:min_len]
    word_annotations_trimmed = word_annotations[:min_len]

    # Create epochs
    epochs = mne.Epochs(raw_subset, word_events, event_id=None, tmin=-3, tmax=3, preload=True, baseline=None, event_repeated='drop')

    # Set metadata
    epochs.metadata = word_annotations_trimmed

    # Resample to 100 Hz
    epochs.resample(500)

    # Define classifier
    clf = make_pipeline(StandardScaler(), Ridge(alpha=1000))
    time_decod = SlidingEstimator(clf, n_jobs=-1, scoring=spearman_scorer, verbose=True)
    cv = KFold(n_splits=5, shuffle=True)

    # Get data
    X = epochs._data

    # Define colors for features
    colors = {
        'Layer_8_PC1': 'navy',
        'Layer_8_PC2': 'darkblue',
        'Layer_8_PC3': 'blue',
        'Layer_8_PC4': 'deepskyblue',
        'Layer_8_PC5': 'lightblue'
    }

    # Loop through features and perform decoding
    all_scores = []
    for feature in ['Layer_8_PC1', 'Layer_8_PC2', 'Layer_8_PC3', 'Layer_8_PC4', 'Layer_8_PC5']:
        y = epochs.metadata[feature].values
        finite_idx = np.isfinite(y)

        # Fit and get scores
        scores = cross_val_multiscore(time_decod, X[finite_idx, ...], y[finite_idx], cv=cv, n_jobs=-1)

        # Store scores for the current range and feature
        all_scores.append((scores, feature, epochs.times))

    # Create a single figure
    fig, ax = plt.subplots(figsize=(14, 10))

    # Iterate over features to plot
    for scores, feature, times in all_scores:
        m = scores.mean(0)
        sem = scores.std(0) / np.sqrt(scores.shape[0])
        ax.plot(times, m, label=feature, color=colors[feature])
        ax.fill_between(times, m - sem, m + sem, alpha=0.2, color=colors[feature])

    # Aesthetics
    ax.axhline(0, color="k", linestyle="--", label="chance")
    ax.set_xlabel("Time (s) relative to word onset")
    ax.set_ylabel("Spearman R")
    ax.legend()
    ax.axvline(0.0, color="k", linestyle="-")
    ax.set_title(f"{analysis_type} - Distribution of Anatomical Sites")
    ax.set_ylim([0, 0.5])  # Set y-axis limit

    # Adjust layout
    plt.tight_layout()

    # Save figure
    plt.savefig(os.path.join(fig_path, f'{analysis_type}_{freq}_{stim}_{run}_gpt2.png'), dpi=200)
    plt.show()

    print(f"Decoding analysis completed for {analysis_type}.")

# Loop through each analysis type and run the analysis
for analysis_type in ANALYSIS_TYPES:
    run_analysis(analysis_type)

print("All analyses completed.")
