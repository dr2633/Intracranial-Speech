import os
import mne
import pandas as pd

freq = '70-150Hz-downsampled'


# Base directory for concatenated files
base_dir = f'/Users/derekrosenzweig/PycharmProjects/CCA-reduce/derivatives/concatenated/{freq}/preprocessed'
output_path = f'/Users/derekrosenzweig/PycharmProjects/CCA-reduce/derivatives/concatenated/{freq}/trial_information'

# Create output directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)

# Initialize list to collect trial information
all_trial_info = []

# Loop through files in the directory
for file_name in os.listdir(base_dir):
    if file_name.endswith('.fif'):
        file_path = os.path.join(base_dir, file_name)
        print(f"Processing {file_path}")

        # Extract stimulus and run information from file name
        parts = file_name.split('_')
        if len(parts) < 4:
            print(f"Filename {file_name} does not conform to expected format, skipping...")
            continue

        stim = parts[0]
        run = parts[1]

        # Load the raw file
        try:
            raw = mne.io.read_raw_fif(file_path, preload=True)

            # Create trial info
            trial_info = pd.DataFrame({
                'Stimulus': [stim],
                'Run': [run],
                'Frequency': ['1-40Hz'],
                'Preprocessed Data Shape': [str(raw.get_data().shape)]
            })
            all_trial_info.append(trial_info)

        except FileNotFoundError:
            print(f"File '{file_path}' not found. Skipping...")
            continue

# Combine all trial info and save to a TSV file
if all_trial_info:
    combined_trial_info = pd.concat(all_trial_info, ignore_index=True)
    trial_info_file = os.path.join(output_path, f'{freq}_trial_info.tsv')
    combined_trial_info.to_csv(trial_info_file, sep='\t', index=False)
    print(f"Combined trial information saved to {trial_info_file}")

print("Trial info creation complete.")

