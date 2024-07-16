import os
import mne
import pandas as pd
import numpy as np

# Define frequency bands and directories
freq_bands = ['1-40Hz', '70-150Hz']
base_input_path = '/Users/derekrosenzweig/PycharmProjects/CCA-reduce/derivatives/individual'
base_output_path = '/Users/derekrosenzweig/PycharmProjects/CCA-reduce/derivatives/individual'

# Target sampling rate
target_fs = 100


def process_file(input_file, output_file):
    raw = mne.io.read_raw_fif(input_file, preload=True)
    if raw.info['sfreq'] > target_fs:
        raw_downsampled = raw.resample(target_fs)
    else:
        raw_downsampled = raw
    raw_downsampled.save(output_file, overwrite=True)
    return raw_downsampled


def create_trial_info(sub, ses, stim, run, freq, raw):
    return pd.DataFrame({
        'Subject': [sub],
        'Session': [ses],
        'Stimulus': [stim],
        'Run': [run],
        'Frequency': [freq],
        'Preprocessed Data': [str(raw.get_data().shape)]
    })


# List of files to loop through
session_stimuli = {
    'ses-01': ['Jobs1', 'Jobs2', 'Jobs3'],
    'ses-02': ['AttFast', 'AttSlow', 'BecFast', 'BecSlow', 'CampFast', 'CampSlow']
}

runs = ['run-01', 'run-02']
subjects = ['sub-01','sub-02','sub-03','sub-04','sub-05','sub-06','sub-07','sub-08']

for freq in freq_bands:
    input_dir = os.path.join(base_input_path, freq, 'preprocessed')
    output_dir = os.path.join(base_output_path, f'{freq}-downsampled', 'preprocessed')
    trial_info_dir = os.path.join(base_output_path, f'{freq}-downsampled', 'trial_information')

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(trial_info_dir, exist_ok=True)

    all_trial_info = []

    for sub in subjects:
        for ses, stims in session_stimuli.items():
            for stim in stims:
                for run in runs:
                    input_file = os.path.join(input_dir, f'{sub}_{ses}_task-listen{stim}_{run}_ieeg.fif')
                    output_file = os.path.join(output_dir, f'{sub}_{ses}_task-listen{stim}_{run}_ieeg.fif')

                    if not os.path.exists(input_file):
                        print(f"File does not exist: {input_file}, skipping...")
                        continue

                    print(f"Processing {input_file}")
                    raw_downsampled = process_file(input_file, output_file)
                    print(f"Saved downsampled file to {output_file}")

                    # Create trial info
                    trial_info = create_trial_info(sub, ses, stim, run, freq, raw_downsampled)
                    all_trial_info.append(trial_info)

    # Combine all trial info and save
    if all_trial_info:
        combined_trial_info = pd.concat(all_trial_info, ignore_index=True)
        trial_info_file = os.path.join(trial_info_dir, f'{freq}_trial_info.tsv')
        combined_trial_info.to_csv(trial_info_file, sep='\t', index=False)
        print(f"Combined trial information saved to {trial_info_file}")

print("Downsampling and trial info creation complete.")