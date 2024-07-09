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
import seaborn as sns

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
word_path = os.path.join(base_path, 'annotations', 'words', 'tsv', f'{stim}-words.tsv')
fig_path = os.path.join(base_path, 'figures', 'Entropy-decoding')
os.makedirs(fig_path, exist_ok=True)

# Load master electrodes file
master_electrodes = pd.read_csv(master_electrodes_path, delimiter='\t')

# Load iEEG data
raw = mne.io.read_raw_fif(file_path, preload=True)
ieeg_data = raw.get_data()

# Load word annotations
word_data = pd.read_csv(word_path, delimiter='\t')

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

# Create quartiles for 'GPT-2 Surprisal' in word annotations
word_data['Surprisal_Quartile'] = pd.qcut(word_data['GPT2_Surprisal'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

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
    word_onsets = (word_data['Start'].values * raw.info['sfreq']).astype(int)

    # Create events for word epochs
    word_events = np.column_stack((word_onsets, np.zeros_like(word_onsets), np.ones_like(word_onsets)))

    # Ensure the number of events matches the number of metadata rows
    min_len = min(len(word_events), len(word_data))
    word_events = word_events[:min_len]
    word_data_trimmed = word_data[:min_len]

    # Create epochs
    epochs = mne.Epochs(raw_subset, word_events, event_id=None, tmin=-0.2, tmax=0.6, preload=True, baseline=None,
                        event_repeated='drop')

    # Set metadata
    epochs.metadata = word_data_trimmed

    # Resample to 100 Hz
    epochs.resample(100)

    # Filter epochs based on quartiles for decoding
    filtered_epochs = [epochs[epochs.metadata['Surprisal_Quartile'] == q] for q in ['Q1', 'Q2', 'Q3', 'Q4']]

    # Create data matrix X and label vector y
    X = np.concatenate([epoch.get_data(copy=True) for epoch in filtered_epochs], axis=0)
    y = np.concatenate([np.ones(len(epoch)) * idx for idx, epoch in enumerate(filtered_epochs)])

    # Plotting
    fig, ax = plt.subplots(1, figsize=(10, 5))
    plt.title(f"Decoding Accuracy for {analysis_type} ({freq})", fontsize=18, fontweight='bold')

    # Preallocate accuracy scores array
    accuracy_scores = np.zeros((4, X.shape[-1]))  # Four quartiles: Q1, Q2, Q3, and Q4

    # Extract data and compute ROC-AUC scores across time
    for tt in range(X.shape[-1]):
        X_t = X[:, :, tt]
        for idx, label in enumerate(['Q1', 'Q2', 'Q3', 'Q4']):
            y_binary = (y == idx).astype(int)
            scores = cross_val_score(clf, X_t, y_binary, scoring='roc_auc', cv=cv, n_jobs=-1)
            accuracy_scores[idx, tt] = scores.mean()

    # Define shades of blue
    blue_palette = sns.color_palette('Blues', 4)  # Four quartiles: Q1, Q2, Q3, Q4

    # Plot accuracy scores with different shades of blue
    for idx, label in enumerate(['Q1', 'Q2', 'Q3', 'Q4']):
        ax.plot(epochs.times, accuracy_scores[idx], label=label, color=blue_palette[idx])

    # Add vertical dashed grey line at t=0
    ax.axvline(x=0, color='grey', linestyle='--')

    # Add horizontal dashed line at y=0.5 (chance level decoding)
    ax.axhline(y=0.5, color='grey', linestyle='--')

    ax.set_xlabel("Time (ms) relative to word onset", fontsize=14)
    ax.set_ylabel("ROC-AUC", fontsize=14)
    ax.legend()
    plt.savefig(os.path.join(fig_path, f'{analysis_type}_{freq}_{stim}_{run}_logistic.jpg'), dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Decoding analysis completed for {analysis_type}.")

# Loop through each analysis type and run the analysis
for analysis_type in ANALYSIS_TYPES:
    run_analysis(analysis_type)

print("All analyses completed.")
