import os
import pandas as pd
import numpy as np
import mne
from sklearn.cross_decomposition import CCA
from scipy.signal import resample_poly
from scipy.stats import zscore

# Parameters
ANALYSIS_TYPE = 'F0'  # Options: 'F0', 'ENTROPY', 'INTENSITY', 'EMBEDDING'
freq = '70-150Hz'  # Options: '1-40Hz', '70-150Hz'
significance_percentile = 90  # Top 10% for significance in both tails
bootstrap_iterations = 5  # Number of bootstrap iterations for p-value computation
use_bootstrap = True  # Set to True to use bootstrapping, False to use standardize-and-threshold

# Paths
base_path = '/Users/derekrosenzweig/PycharmProjects/CCA-reduce'
fif_file_path = f'{base_path}/derivatives/combined/{freq}/{freq}_master_raw.fif'
start_times_csv_path = f'{base_path}/derivatives/combined/{freq}/{freq}_start_times.csv'
annotations_dir = os.path.join(base_path, 'annotations', 'acoustics', 'tsv')
entropy_dir = os.path.join(base_path, 'annotations', 'surprisal', 'tsv')
embedding_dir = os.path.join(base_path, 'annotations', 'embeddings', 'tsv')
electrode_info_file_path = os.path.join(base_path, 'electrodes', 'master-electrodes.tsv')
csv_directory = os.path.join(base_path, 'cca-weights', 'weights-bootstrap')
correlation_output_path = os.path.join(base_path, 'cca-weights', 'weights-bootstrap', 'corr_coef', f'{ANALYSIS_TYPE}_{freq}_corr_coeff.tsv')
os.makedirs(csv_directory, exist_ok=True)

# Load a subset of the concatenated FIF file
raw = mne.io.read_raw_fif(fif_file_path, preload=True)
raw.crop(tmax=3000)
ieeg_data = raw.get_data()

# Load electrode information
electrode_info = pd.read_csv(electrode_info_file_path, delimiter='\t')

# Ensure the 'Exclude' column exists and fill NaNs
electrode_info['Exclude'] = electrode_info['Exclude'].fillna('')

# In 'Anat', exclude all electrodes with the string 'Unknown', 'Left-Cerebral-White-Matter', 'Right-Cerebral-White-Matter', or 'undefined'
exclude_anat = ['Unknown', 'Left-Cerebral-White-Matter', 'Right-Cerebral-White-Matter', 'undefined']
exclude_electrodes_anat = electrode_info[electrode_info['Anat'].isin(exclude_anat)]['sub_name'].values

# Create a mapping from current channel names to 'sub_name'
channel_mapping = {raw.ch_names[i]: electrode_info.iloc[i]['sub_name'] for i in range(len(raw.ch_names))}

# Rename channels in the raw data
raw.rename_channels(channel_mapping)

# Identify electrodes to exclude
exclude_electrodes = np.concatenate((
    electrode_info.loc[electrode_info['Exclude'] == 'Yes', 'sub_name'].values,
    exclude_electrodes_anat
))

# Get the indices of the electrodes to exclude
exclude_indices = [raw.ch_names.index(electrode) for electrode in exclude_electrodes if electrode in raw.ch_names]

# Create a mask to exclude the electrodes
include_indices = [i for i in range(len(raw.ch_names)) if i not in exclude_indices]

# Apply the mask to filter the ieeg_data
ieeg_data_filtered = ieeg_data[include_indices, :]

# Create a new RawArray with the filtered data
info = mne.create_info([raw.ch_names[i] for i in include_indices], raw.info['sfreq'], ch_types='eeg')
raw_filtered = mne.io.RawArray(ieeg_data_filtered, info)

# Downsample the iEEG data to match the annotation data sampling rate
target_fs = 100
original_fs = raw_filtered.info['sfreq']
downsampling_factor = int(original_fs / target_fs)
ieeg_data_downsampled = resample_poly(ieeg_data_filtered, up=1, down=downsampling_factor, axis=1)

# Load start times
start_times_df = pd.read_csv(start_times_csv_path)

# Initialize list to store concatenated annotations
all_annotation_data = []

# Process each stimulus and adjust annotations
for index, row in start_times_df.iterrows():
    stim = row['Filename'].split('_')[0]
    start_sample = row['StartSample']

    if ANALYSIS_TYPE == 'F0':
        annotation_file = os.path.join(annotations_dir, f'{stim}-acoustics.tsv')
        column_name = 'F0 (Hz)'
    elif ANALYSIS_TYPE == 'ENTROPY':
        annotation_file = os.path.join(entropy_dir, f'{stim}-words.tsv')
        column_name = 'GPT2_Surprisal'
    elif ANALYSIS_TYPE == 'INTENSITY':
        annotation_file = os.path.join(annotations_dir, f'{stim}-acoustics.tsv')
        column_name = 'Intensity (dB)'
    elif ANALYSIS_TYPE == 'EMBEDDING_1':
        annotation_file = os.path.join(embedding_dir, f'{stim}-words.tsv')
        column_name = 'Layer_8_PC1'
    elif ANALYSIS_TYPE == 'EMBEDDING_2':
        annotation_file = os.path.join(embedding_dir, f'{stim}-words.tsv')
        column_name = 'Layer_8_PC2'
    elif ANALYSIS_TYPE == 'EMBEDDING_3':
        annotation_file = os.path.join(embedding_dir, f'{stim}-words.tsv')
        column_name = 'Layer_8_PC3'
    elif ANALYSIS_TYPE == 'EMBEDDING_4':
        annotation_file = os.path.join(embedding_dir, f'{stim}-words.tsv')
        column_name = 'Layer_8_PC4'
    elif ANALYSIS_TYPE == 'EMBEDDING_5':
        annotation_file = os.path.join(embedding_dir, f'{stim}-words.tsv')
        column_name = 'Layer_8_PC5'

    if os.path.exists(annotation_file):
        # Load annotation data
        annotation_data = pd.read_csv(annotation_file, delimiter='\t')
        annotation_values = annotation_data[column_name].fillna(0).values
        annotation_transposed = annotation_values.reshape(-1, 1)

        # Adjust annotation data to the cumulative time
        adjusted_annotation = np.zeros((start_sample + len(annotation_transposed), 1))
        adjusted_annotation[start_sample:start_sample + len(annotation_transposed)] = annotation_transposed

        # Append to the master annotation data list
        all_annotation_data.append(adjusted_annotation[start_sample:])

# Concatenate all annotation data
if all_annotation_data:
    all_annotation_data = np.concatenate(all_annotation_data, axis=0)

    # Ensure annotation data matches the iEEG data length
    if all_annotation_data.shape[0] < ieeg_data_downsampled.shape[1]:
        all_annotation_data = np.pad(all_annotation_data,
                                     ((0, ieeg_data_downsampled.shape[1] - all_annotation_data.shape[0]), (0, 0)),
                                     'constant')
    elif all_annotation_data.shape[0] > ieeg_data_downsampled.shape[1]:
        all_annotation_data = all_annotation_data[:ieeg_data_downsampled.shape[1]]

    # Perform CCA
    ieeg_data_transposed = ieeg_data_downsampled.T
    n_components = 1
    cca = CCA(n_components=n_components)
    cca.fit(ieeg_data_transposed, all_annotation_data)
    X_transformed, Y_transformed = cca.transform(ieeg_data_transposed, all_annotation_data)

    # Obtain CCA weights and compute the canonical correlation
    ieeg_weights = cca.x_loadings_[:, 0]
    canonical_correlation = np.corrcoef(X_transformed.T, Y_transformed.T)[0, 1]
    print(f"Canonical Correlations for {ANALYSIS_TYPE} at {freq}:", canonical_correlation)

    if use_bootstrap:
        # Bootstrap resampling for significance testing
        bootstrap_weights = np.zeros((bootstrap_iterations, len(ieeg_weights)))
        for i in range(bootstrap_iterations):
            sample_indices = np.random.choice(ieeg_data_transposed.shape[0], ieeg_data_transposed.shape[0],
                                              replace=True)
            cca.fit(ieeg_data_transposed[sample_indices], all_annotation_data[sample_indices])
            bootstrap_weights[i] = cca.x_loadings_[:, 0]

        # Compute p-values based on the bootstrap distribution
        p_values = np.mean(np.abs(bootstrap_weights) >= np.abs(ieeg_weights), axis=0)
    else:
        # Standardize the weights using z-scores
        standardized_weights = zscore(ieeg_weights)
        # Determine the significance threshold
        threshold = np.percentile(np.abs(standardized_weights), significance_percentile)
        significant_indices = np.where(np.abs(standardized_weights) >= threshold)[0]

    # Save the canonical correlation coefficient and p-value
    correlation_data = pd.DataFrame({
        'AnalysisType': [ANALYSIS_TYPE],
        'FrequencyRange': [freq],
        'CanonicalCorrelation': [canonical_correlation],
        'SignificantElectrodes': [len(significant_indices)] if not use_bootstrap else ['NA'],
        'BootstrapSamples': [bootstrap_iterations if use_bootstrap else 'NA']
    })
    correlation_data.to_csv(correlation_output_path, sep='\t', index=False, mode='a',
                            header=not os.path.exists(correlation_output_path))

    # Filter electrode_info to match the filtered raw data
    electrode_info_filtered = electrode_info[electrode_info['sub_name'].isin(raw_filtered.ch_names)]

    # Ensure the filtered electrode info matches the length of the raw_filtered data
    electrode_info_filtered = electrode_info_filtered.set_index('sub_name').loc[raw_filtered.ch_names].reset_index()

    # Integrate custom channel info and weights
    weights_df = pd.DataFrame({
        'ch_name': raw_filtered.ch_names,
        'weight': ieeg_weights,
        'p_value' if use_bootstrap else 'z_score': p_values if use_bootstrap else standardized_weights
    })
    combined_df = pd.merge(weights_df, electrode_info_filtered, left_on='ch_name', right_on='sub_name', how='inner')

    # Sort the DataFrame based on the absolute values of weights
    sorted_df = combined_df.sort_values(by='weight', key=abs, ascending=False)

    # Save the sorted DataFrame to CSV
    csv_filename = os.path.join(csv_directory, f'{freq}_CCA_weights_{ANALYSIS_TYPE}.csv')
    sorted_df.to_csv(csv_filename, index=False, columns=[
        'ch_name', 'weight', 'p_value' if use_bootstrap else 'z_score', 'x', 'y', 'z', 'name', 'sub_name', 'WMvsGM',
        'LvsR', 'Anat', 'Desikan_Killiany',
        'fsaverageINF_coord_1', 'fsaverageINF_coord_2', 'fsaverageINF_coord_3',
        'ScannerNativeRAS_coord_1', 'ScannerNativeRAS_coord_2', 'ScannerNativeRAS_coord_3',
        'MGRID_coord_1', 'MGRID_coord_2', 'MGRID_coord_3',
        'subINF_coord_1', 'subINF_coord_2', 'subINF_coord_3'
    ])

    print(f"Enhanced ordered list of electrodes with channel info saved to {csv_filename}")
    print(f"Canonical correlation coefficient and significant electrodes count saved to {correlation_output_path}")

