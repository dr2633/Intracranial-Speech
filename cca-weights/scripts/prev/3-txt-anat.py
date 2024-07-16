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

# List of anatomical sites to process
anatomical_sites = [
'Insula',
'Thalamus',
'Inferior Frontal Gyrus (IFG)'

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

        for site in anatomical_sites:
            site_data = electrode_data[electrode_data['Anat'] == site]

            # Split electrode data based on hemisphere
            lh_data = site_data[site_data['LvsR'] == 'L']
            rh_data = site_data[site_data['LvsR'] == 'R']

            # Prepare the output directory and file paths
            site_dir = os.path.join(output_dir, site.replace(' ', '_').replace('(', '').replace(')', ''))
            os.makedirs(site_dir, exist_ok=True)

            lh_file = os.path.join(site_dir, f'{site.replace(" ", "_").replace("(", "").replace(")", "")}_{analysis_type}_{freq}_lh.txt')
            rh_file = os.path.join(site_dir, f'{site.replace(" ", "_").replace("(", "").replace(")", "")}_{analysis_type}_{freq}_rh.txt')

            # Save LH and RH data to separate files without headers
            lh_data[['fsaverageINF_coord_1', 'fsaverageINF_coord_2', 'fsaverageINF_coord_3', 'LvsR', 'weight']].to_csv(lh_file, sep=' ', index=False, header=False, quoting=3)
            rh_data[['fsaverageINF_coord_1', 'fsaverageINF_coord_2', 'fsaverageINF_coord_3', 'LvsR', 'weight']].to_csv(rh_file, sep=' ', index=False, header=False, quoting=3)

            print(f"Saved {lh_file} and {rh_file}")

    except Exception as e:
        print(f"Error processing {input_file_path}: {e}")
