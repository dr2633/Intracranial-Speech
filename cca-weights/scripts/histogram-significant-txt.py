import os
import pandas as pd
import scipy.stats as stats
import numpy as np

# Define the paths
weights_path = '/Users/derekrosenzweig/PycharmProjects/CCA-reduce/cca-weights/weights'
output_dir = '/Users/derekrosenzweig/PycharmProjects/CCA-reduce/cca-weights/txt/txt-significant'
lh_output_dir = os.path.join(output_dir, 'lh')
rh_output_dir = os.path.join(output_dir, 'rh')
os.makedirs(lh_output_dir, exist_ok=True)
os.makedirs(rh_output_dir, exist_ok=True)

# Parameters
z_threshold = 2  # Z-score threshold for significance
files_and_titles = [
    'ENTROPY',  # for filepath input 'ENTROPY'
    'F0',  # 'F0'
    'INTENSITY',  # INTENSITY
    'EMBEDDING_1',  # EMBEDDING
    'EMBEDDING_2',
    'EMBEDDING_3',
    'EMBEDDING_4',
    'EMBEDDING_5'
]
freq = '70-150Hz'

# Debug: List all files in the weights_path directory
print("Files in the directory:")
print(os.listdir(weights_path))

# Convert CSV Files to TXT Files
for analysis_type in files_and_titles:
    # Construct the input file path
    input_file_path = os.path.join(weights_path, f'{freq}_CCA_weights_{analysis_type}.csv')

    # Check if the file exists
    if not os.path.exists(input_file_path):
        print(f"File does not exist: {input_file_path}")
        continue

    # Load electrode data from CSV
    try:
        column_names = ['ch_name', 'weight', 'x', 'y', 'z', 'name', 'sub_name', 'WMvsGM', 'LvsR', 'Anat', 'Desikan_Killiany',
                        'fsaverageINF_coord_1', 'fsaverageINF_coord_2', 'fsaverageINF_coord_3',
                        'ScannerNativeRAS_coord_1', 'ScannerNativeRAS_coord_2', 'ScannerNativeRAS_coord_3',
                        'MGRID_coord_1', 'MGRID_coord_2', 'MGRID_coord_3',
                        'subINF_coord_1', 'subINF_coord_2', 'subINF_coord_3']
        electrode_data = pd.read_csv(input_file_path, names=column_names, header=0)

        print(f"Loaded {input_file_path} with shape: {electrode_data.shape}")
        print(electrode_data.head())

        # Calculate z-scores for weights
        z_scores = stats.zscore(electrode_data['weight'])
        electrode_data['z_score'] = z_scores

        # Filter for significant electrodes
        significant_data = electrode_data[np.abs(electrode_data['z_score']) > z_threshold]

        # Split electrode data based on hemisphere
        lh_data = significant_data[significant_data['LvsR'] == 'L']
        rh_data = significant_data[significant_data['LvsR'] == 'R']

        # Prepare the output file paths
        lh_file = os.path.join(lh_output_dir, f'significant_{analysis_type}_{freq}_lh.txt')
        rh_file = os.path.join(rh_output_dir, f'significant_{analysis_type}_{freq}_rh.txt')

        # Save split data to separate files without headers
        lh_data[['fsaverageINF_coord_1', 'fsaverageINF_coord_2', 'fsaverageINF_coord_3', 'LvsR', 'weight', 'WMvsGM']].to_csv(lh_file, sep=' ', index=False, header=False)
        rh_data[['fsaverageINF_coord_1', 'fsaverageINF_coord_2', 'fsaverageINF_coord_3', 'LvsR', 'weight', 'WMvsGM']].to_csv(rh_file, sep=' ', index=False, header=False)

        print(f"Saved {lh_file} and {rh_file}")

    except Exception as e:
        print(f"Error processing {input_file_path}: {e}")

print("Finished processing all files.")
