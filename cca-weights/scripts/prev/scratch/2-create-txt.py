import os
import pandas as pd
import re

base_path = '/Users/derekrosenzweig/PycharmProjects/CCA-reduce'


weights_path = '/cca-weights/weights'


output_dir = '/cca-weights/txt'

n = 1000

files_and_titles = [
    'ENTROPY',  # for filepath input 'ENTROPY'
    'F0',  # 'F0'
    'INTENSITY',  # INTENSITY
    'EMBEDDING',  # EMBEDDING
    'EMBEDDING_2',
    'EMBEDDING_3',
    'EMBEDDING_4',
    'EMBEDDING_5'
]

freq = '70-150Hz'

# Regex to split lines considering quotes and spaces
pattern = re.compile(r'\".*?\"|\S+')

for analysis_type in files_and_titles:
    # Electrode file
    electrode_file = f'{input_dir}/top_{n}_electrodes_{analysis_type}_{freq}.txt'

    # Load electrode data
    try:
        # Read file content
        with open(electrode_file, 'r') as file:
            lines = file.readlines()

        # Process each line using the regex
        processed_data = []
        for line in lines:
            matches = pattern.findall(line)
            if len(matches) == 7:
                x, y, z, sub_name, unknown, LvsR, weight = matches
                processed_data.append([float(x), float(y), float(z), LvsR, float(weight)])
            elif len(matches) == 6:
                # Handle case where unknown column might be missing
                x, y, z, LvsR, weight = matches[:3] + matches[4:]
                processed_data.append([float(x), float(y), float(z), LvsR, float(weight)])

        # Convert to DataFrame
        electrode_data = pd.DataFrame(processed_data, columns=['x', 'y', 'z', 'LvsR', 'weight'])

        print(f"Loaded {electrode_file} with shape: {electrode_data.shape}")
        print(electrode_data.head())

        # Split electrode data based on hemisphere
        lh_data = electrode_data[electrode_data['LvsR'] == 'L']
        rh_data = electrode_data[electrode_data['LvsR'] == 'R']

        # Save split data to separate files
        lh_file = f'{output_dir}/top_{n}_electrodes_{analysis_type}_{freq}_lh.txt'
        rh_file = f'{output_dir}/top_{n}_electrodes_{analysis_type}_{freq}_rh.txt'

        lh_data.to_csv(lh_file, sep=' ', index=False, header=False)
        rh_data.to_csv(rh_file, sep=' ', index=False, header=False)

        print(f"Saved {lh_file} and {rh_file}")

    except Exception as e:
        print(f"Error processing {electrode_file}: {e}")
