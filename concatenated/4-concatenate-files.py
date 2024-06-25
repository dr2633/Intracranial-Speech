import mne
import numpy as np
import os

# List all sessions, stimuli, and runs to loop through
session_stimuli = {
    'ses-01': ['Jobs1', 'Jobs2', 'Jobs3'],
    'ses-02': ['AttFast', 'AttSlow', 'BecFast', 'BecSlow', 'CampFast', 'CampSlow']
}
runs = ['run-01', 'run-02']

# Set parameters for path
freq = '1-40Hz'
pro = 'preprocessed'
base_path = f'/Users/derekrosenzweig/Documents/GitHub/iEEG-Speech/derivatives/individual/{freq}/{pro}'

# Iterate through sessions, stimuli, and runs
for ses, stimuli in session_stimuli.items():
    for stim in stimuli:
        for run in runs:
            # Create an empty list to store file paths
            file_paths = []

            # Create a for loop that gets files for sub = 'sub-01' through 'sub-08', excluding sub-04
            for i in range(2, 9):
                if i == 4:
                    continue
                sub = f'sub-{i:02d}'
                fif_file = f'{base_path}/{sub}_{ses}_task-listen{stim}_{run}_ieeg.fif'
                if os.path.exists(fif_file):
                    file_paths.append(fif_file)

            # Append sub-04 if it exists (session 1)
            sub_04_file = f'{base_path}/sub-04_{ses}_task-listen{stim}_{run}_ieeg.fif'
            if os.path.exists(sub_04_file):
                file_paths.append(sub_04_file)

            # Create an empty list to store the raw data from each file
            raw_data_list = []

            # Find the maximum number of time points across all files
            max_time_points = 0

            # Read raw data from each file and append to the list
            for file_path in file_paths:
                raw = mne.io.read_raw_fif(file_path)
                raw_data_list.append(raw)
                max_time_points = max(max_time_points, raw.n_times)

            # Create a new data array with the combined number of channels and maximum time points
            combined_data = np.zeros((sum(raw.info['nchan'] for raw in raw_data_list), max_time_points))

            # Create a list to store the combined channel names
            combined_ch_names = []

            # Concatenate the data and zero-pad shorter files
            start_index = 0
            for raw in raw_data_list:
                data = raw.get_data()
                num_channels, num_time_points = data.shape
                combined_data[start_index:start_index + num_channels, :num_time_points] = data
                combined_ch_names.extend(raw.ch_names)
                start_index += num_channels

            # Create a new Info object with the combined channel names
            info = mne.create_info(combined_ch_names, raw_data_list[0].info['sfreq'], ch_types='seeg')

            # Create a new Raw object with the concatenated data
            concatenated_raw = mne.io.RawArray(combined_data, info)

            # Save concatenated raw data to a new fif file
            save_path = f'/Users/derekrosenzweig/Documents/GitHub/iEEG-Speech/derivatives/concatenated/{freq}/{pro}'
            os.makedirs(save_path, exist_ok=True)  # Create the directory if it doesn't exist
            concatenated_file = f'{save_path}/{stim}_{run}_concatenated_raw.fif'
            concatenated_raw.save(concatenated_file, overwrite=True)

            print(
                f"Concatenation complete for stimulus {stim}, session {ses}, and run {run}. Raw data saved to: {concatenated_file}")
