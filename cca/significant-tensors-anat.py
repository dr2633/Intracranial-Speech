import os
import pandas as pd
import scipy.stats as stats
import numpy as np
import mne
import torch

# Paths
base_path = '/Users/derekrosenzweig/PycharmProjects/CCA-reduce'
weights_path = os.path.join(base_path, 'cca-weights', 'weights')
master_electrodes_path = os.path.join(base_path, 'electrodes', 'master-electrodes.tsv')
file_path = os.path.join(base_path, 'derivatives', 'concatenated', '70-150Hz', 'preprocessed', 'Jobs1_run-01_concatenated_raw.fif')

# Load master electrodes file
master_electrodes = pd.read_csv(master_electrodes_path, delimiter='\t')

# Load iEEG data
raw = mne.io.read_raw_fif(file_path, preload=True)
ieeg_data = raw.get_data()

# Set the channel names in raw iEEG data to sub_name values from master electrodes
channel_mapping = {raw.ch_names[i]: master_electrodes['sub_name'].iloc[i] for i in range(len(raw.ch_names))}
raw.rename_channels(channel_mapping)

# List of analysis types
analysis_types = ['F0', 'ENTROPY', 'INTENSITY', 'EMBEDDING_1', 'EMBEDDING_2', 'EMBEDDING_3', 'EMBEDDING_4', 'EMBEDDING_5']

# Function to get torch tensors for significant electrodes
def get_significant_tensors(analysis_type):
    weights_file_path = os.path.join(weights_path, f'70-150Hz_CCA_weights_{analysis_type}.csv')

    # Load weights data
    weights_data = pd.read_csv(weights_file_path)

    # Get significant electrodes
    z_scores = stats.zscore(weights_data['weight'])
    weights_data['z_score'] = z_scores
    significant_data = weights_data[np.abs(weights_data['z_score']) > 2]

    # Create clusters of significant electrodes for the top five electrodes in each anatomical site
    STG_data = weights_data[weights_data['Anatomical-Final'] == 'STG']
    HG_data =  weights_data[weights_data['Anatomical-Final'] == 'HG']



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
for analysis_type in analysis_types:
    tensor = get_significant_tensors(analysis_type)
    tensors_dict[analysis_type] = tensor
    print(f"Shape of tensor for {analysis_type}: {tensor.shape}")

print("Finished creating torch tensors for each feature set.")
