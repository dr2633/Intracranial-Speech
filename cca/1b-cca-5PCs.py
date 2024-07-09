import os
import pandas as pd
import numpy as np
import mne
from sklearn.cross_decomposition import CCA
from scipy.signal import resample_poly
from scipy.stats import f
from sklearn.model_selection import KFold

# Parameters
ANALYSIS_TYPES = ['EMBEDDING_1', 'EMBEDDING_2', 'EMBEDDING_3', 'EMBEDDING_4', 'EMBEDDING_5']  # Options: 'F0', 'ENTROPY', 'INTENSITY', 'EMBEDDING'
freq = '70-150Hz'  # Options: '1-40Hz', '70-150Hz'
n_folds = 5  # Number of cross-validation folds

# Paths
base_path = '/Users/derekrosenzweig/PycharmProjects/CCA-reduce'
fif_file_path = f'{base_path}/derivatives/combined/{freq}/{freq}_master_raw.fif'
start_times_csv_path = f'{base_path}/derivatives/combined/{freq}/{freq}_start_times.csv'
annotations_dir = os.path.join(base_path, 'annotations', 'acoustics', 'tsv')
entropy_dir = os.path.join(base_path, 'annotations', 'surprisal', 'tsv')
embedding_dir = os.path.join(base_path, 'annotations', 'embeddings', 'tsv')
electrode_info_file_path = os.path.join(base_path, 'electrodes', 'master-electrodes.tsv')
csv_directory = os.path.join(base_path, 'cca-weights', 'weights')
correlation_output_path = os.path.join(base_path, 'cca-weights', 'corr_coef', f'5-PCs_corr_coeff.tsv')
os.makedirs(csv_directory, exist_ok=True)
os.makedirs(os.path.dirname(correlation_output_path), exist_ok=True)

# Load a subset of the concatenated FIF file
raw = mne.io.read_raw_fif(fif_file_path, preload=True)
raw.crop(tmax=3000)
ieeg_data = raw.get_data()

# Load electrode information
electrode_info = pd.read_csv(electrode_info_file_path, delimiter='\t')

# Ensure the 'Exclude' column exists and fill NaNs
electrode_info['Exclude'] = electrode_info['Exclude'].fillna('')

# In 'Anat', exclude all electrodes with the string 'Unknown', , or 'undefined'
exclude_anat = ['Unknown', 'undefined']
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
    combined_annotations = None

    for analysis_type in ANALYSIS_TYPES:
        if 'EMBEDDING' in analysis_type:
            annotation_file = os.path.join(embedding_dir, f'{stim}-words.tsv')
            column_name = f'Layer_8_PC{ANALYSIS_TYPES.index(analysis_type) + 1}'

            if os.path.exists(annotation_file):
                # Load annotation data
                annotation_data = pd.read_csv(annotation_file, delimiter='\t')
                annotation_values = annotation_data[column_name].fillna(0).values
                annotation_transposed = annotation_values.reshape(-1, 1)

                # Adjust annotation data to the cumulative time
                adjusted_annotation = np.zeros((start_sample + len(annotation_transposed), 1))
                adjusted_annotation[start_sample:start_sample + len(annotation_transposed)] = annotation_transposed

                if combined_annotations is None:
                    combined_annotations = adjusted_annotation[start_sample:]
                else:
                    combined_annotations = np.hstack((combined_annotations, adjusted_annotation[start_sample:]))

    if combined_annotations is not None:
        # Append to the master annotation data list
        all_annotation_data.append(combined_annotations)

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

    # Initialize KFold cross-validation
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    canonical_correlations = []
    p_values = []

    # Perform cross-validation
    for train_index, test_index in kf.split(ieeg_data_downsampled.T):
        X_train, X_test = ieeg_data_downsampled[:, train_index].T, ieeg_data_downsampled[:, test_index].T
        Y_train, Y_test = all_annotation_data[train_index], all_annotation_data[test_index]

        # Perform CCA
        cca = CCA(n_components=5)
        cca.fit(X_train, Y_train)
        X_train_transformed, Y_train_transformed = cca.transform(X_train, Y_train)
        X_test_transformed, Y_test_transformed = cca.transform(X_test, Y_test)

        # Compute canonical correlations for each component
        for i in range(5):
            canonical_correlation = np.corrcoef(X_test_transformed[:, i], Y_test_transformed[:, i])[0, 1]
            canonical_correlations.append(canonical_correlation)

            # Compute Wilks' Lambda and p-value
            n_samples = X_test.shape[0]
            n_features_X = X_test.shape[1]
            n_features_Y = Y_test.shape[1]
            wilks_lambda = np.prod(1 - canonical_correlation**2)
            approx_stat = -(n_samples - 1 - (n_features_X + n_features_Y + 1) / 2) * np.log(wilks_lambda)
            df1 = n_features_X * n_features_Y
            df2 = n_samples - 1 - (n_features_X + n_features_Y + 1) / 2
            p_value = f.sf(approx_stat, df1, df2)
            p_values.append(p_value)

    # Average results from cross-validation
    mean_canonical_correlations = np.mean(canonical_correlations, axis=0)
    mean_p_values = np.mean(p_values, axis=0)

    print(f"Mean Canonical Correlations for {ANALYSIS_TYPES} at {freq}:", mean_canonical_correlations)
    print(f"Mean P-value for Canonical Correlation: {mean_p_values}")

    # Save the mean canonical correlation and mean p-value
    correlation_data = pd.DataFrame({
        'AnalysisType': [ANALYSIS_TYPES[0]],
        'FrequencyRange': [freq],
        'MeanCanonicalCorrelation': [mean_canonical_correlations],
        'MeanPValue': [mean_p_values]
    })
    correlation_data.to_csv(correlation_output_path, sep='\t', index=False, mode='a',
                            header=not os.path.exists(correlation_output_path))

    # Fit CCA on the entire dataset to get the weights
    cca.fit(ieeg_data_downsampled.T, all_annotation_data)
    ieeg_weights = cca.x_loadings_

    # Filter electrode_info to match the filtered raw data
    electrode_info_filtered = electrode_info[electrode_info['sub_name'].isin(raw_filtered.ch_names)]

    # Ensure the filtered electrode info matches the length of the raw_filtered data
    electrode_info_filtered = electrode_info_filtered.set_index('sub_name').loc[raw_filtered.ch_names].reset_index()

    # Integrate custom channel info and weights
    weights_df = pd.DataFrame(ieeg_weights, columns=[f'weight_PC{i+1}' for i in range(ieeg_weights.shape[1])])
    weights_df.insert(0, 'ch_name', raw_filtered.ch_names)
    combined_df = pd.merge(weights_df, electrode_info_filtered, left_on='ch_name', right_on='sub_name', how='inner')

    # Sort the DataFrame based on the absolute values of the first component weights
    sorted_df = combined_df.sort_values(by='weight_PC1', key=abs, ascending=False)

    # Save the sorted DataFrame to CSV
    csv_filename = os.path.join(csv_directory, f'{freq}_CCA_weights_5-PCs.csv')
    sorted_df.to_csv(csv_filename, index=False, columns=[
        'ch_name', 'weight_PC1', 'weight_PC2', 'weight_PC3', 'weight_PC4', 'weight_PC5', 'x', 'y', 'z', 'name', 'sub_name', 'WMvsGM', 'LvsR', 'Anat', 'Desikan_Killiany',
        'fsaverageINF_coord_1', 'fsaverageINF_coord_2', 'fsaverageINF_coord_3',
        'ScannerNativeRAS_coord_1', 'ScannerNativeRAS_coord_2', 'ScannerNativeRAS_coord_3',
        'MGRID_coord_1', 'MGRID_coord_2', 'MGRID_coord_3',
        'subINF_coord_1', 'subINF_coord_2', 'subINF_coord_3'
    ])

    print(f"Enhanced ordered list of electrodes with channel info saved to {csv_filename}")
    print(f"Mean canonical correlation coefficient and mean p-value saved to {correlation_output_path}")
else:
    print("No valid annotation data found for the selected analysis type and stimuli.")
