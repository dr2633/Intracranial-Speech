import os
import pandas as pd

base_path = '/'
weights_path = os.path.join(base_path, 'cca-weights', 'weights')
output_dir = os.path.join(base_path, 'cca-weights', 'txt')
os.makedirs(output_dir, exist_ok=True)

# Parameters
n = 1000
freq = '70-150Hz'
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

# Step 1: Convert CSV Files to TXT Files
for analysis_type in files_and_titles:
    # Construct the input file path
    input_file_path = f'{weights_path}/{freq}_CCA_weights_{analysis_type}.csv'

    # Load electrode data from CSV
    try:
        electrode_data = pd.read_csv(input_file_path)

        print(f"Loaded {input_file_path} with shape: {electrode_data.shape}")
        print(electrode_data.head())

        # Split electrode data based on hemisphere
        lh_data = electrode_data[electrode_data['LvsR'] == 'L']
        rh_data = electrode_data[electrode_data['LvsR'] == 'R']

        # Prepare the output file paths
        lh_file = f'{output_dir}/top_{n}_electrodes_{analysis_type}_{freq}_lh.txt'
        rh_file = f'{output_dir}/top_{n}_electrodes_{analysis_type}_{freq}_rh.txt'

        # Save split data to separate files without headers
        lh_data[['fsaverageINF_coord_1', 'fsaverageINF_coord_2', 'fsaverageINF_coord_3', 'LvsR', 'weight', 'WMvsGM']].to_csv(lh_file, sep=' ', index=False, header=False, quoting=3)
        rh_data[['fsaverageINF_coord_1', 'fsaverageINF_coord_2', 'fsaverageINF_coord_3', 'LvsR', 'weight', 'WMvsGM']].to_csv(rh_file, sep=' ', index=False, header=False, quoting=3)

        print(f"Saved {lh_file} and {rh_file}")

    except Exception as e:
        print(f"Error processing {input_file_path}: {e}")

print("Finished processing all files.")
