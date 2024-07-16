import mne
import numpy as np
import os

# Base path for data
base_path = '/Users/derekrosenzweig/PycharmProjects/CCA-reduce'

# List all sessions, stimuli, and runs to loop through
session_stimuli = {
    'ses-01': ['Jobs1', 'Jobs2', 'Jobs3'],
    'ses-02': ['AttFast', 'AttSlow', 'BecFast', 'BecSlow', 'CampFast', 'CampSlow']
}
runs = ['run-01', 'run-02']
freq = '1-40Hz-downsampled'
pro = 'preprocessed'

# Function to print and update channel names
def process_channels(file_paths):
    for session_files in file_paths:
        for sub, file_path in session_files:
            try:
                raw = mne.io.read_raw_fif(file_path)
                for idx, channel in enumerate(raw.ch_names):
                    raw.rename_channels({channel: f"{sub} {channel}"})
                print(f"Channel names in file '{file_path}' with subject '{sub}' updated.")
                print(raw.ch_names)
            except FileNotFoundError:
                print(f"File '{file_path}' not found. Skipping...")
                continue
    print("Channel name updating complete.")

# Function to concatenate data across participants
def concatenate_data(file_paths, stim, ses, run):
    raw_data_list = []
    max_time_points = 0

    # Read raw data from each file and append to the list
    for file_path in file_paths:
        try:
            raw = mne.io.read_raw_fif(file_path)
            raw_data_list.append(raw)
            max_time_points = max(max_time_points, raw.n_times)
        except FileNotFoundError:
            print(f"File '{file_path}' not found. Skipping...")
            continue

    # Check if there are any valid files
    if not raw_data_list:
        print(f"No valid files found for stimulus {stim}, session {ses}, and run {run}. Skipping concatenation.")
        return

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
    save_path = f'{base_path}/derivatives/concatenated/{freq}/{pro}'
    os.makedirs(save_path, exist_ok=True)  # Create the directory if it doesn't exist
    concatenated_file = f'{save_path}/{stim}_{run}_concatenated_raw.fif'
    concatenated_raw.save(concatenated_file, overwrite=True)

    print(f"Concatenation complete for stimulus {stim}, session {ses}, and run {run}. Raw data saved to: {concatenated_file}")

# Main script
def main():
    file_paths = []

    # Iterate through sessions, stimuli, and runs
    for ses, stimuli in session_stimuli.items():
        for stim in stimuli:
            for run in runs:
                session_file_paths = []
                for i in range(2, 9):  # Iterate through sub-02 to sub-08
                    # if i == 4 and ses == 'ses-02':
                    if i == 4:
                        continue
                    sub = f'sub-{i:02d}'
                    fif_file = f'{base_path}/derivatives/individual/{freq}/{pro}/{sub}_{ses}_task-listen{stim}_{run}_ieeg.fif'
                    session_file_paths.append((sub, fif_file))  # Store subject name and file path as tuple
                file_paths.append(session_file_paths)  # Store file paths for the current combination

                # Process channels for the current session, stimulus, and run
                process_channels([session_file_paths])

                # Concatenate data for the current session, stimulus, and run
                concatenate_data([f[1] for f in session_file_paths], stim, ses, run)

if __name__ == "__main__":
    main()
