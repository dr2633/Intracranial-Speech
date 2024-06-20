import mne

# List all sessions, stimuli, and runs to loop through
session_stimuli = {
    'ses-01': ['Jobs1', 'Jobs2', 'Jobs3'],
    'ses-02': ['AttFast', 'AttSlow', 'BecFast', 'BecSlow', 'CampFast', 'CampSlow']
}
runs = ['run-01', 'run-02']

# Set parameters for path
freq = '1-40Hz'
pro = 'preprocessed'
base_path = '/Users/derekrosenzweig/Documents/GitHub/iEEG-Speech/derivatives/individual/1-40Hz/preprocessed'

# Create an empty list to store file paths
file_paths = []

# Iterate through sessions, stimuli, and runs
for ses, stimuli in session_stimuli.items():
    for stim in stimuli:
        for run in runs:
            # Create an empty list to store file paths for the current combination
            session_file_paths = []
            # Create a for loop that gets files for sub = 'sub-01' through 'sub-07'
            for i in range(2, 9):  # Iterate through sub-01 to sub-07
                if i == 4 and ses == 'ses-02':
                    continue
                sub = f'sub-{i:02d}'
                fif_file = f'{base_path}/{sub}_{ses}_task-listen{stim}_{run}_ieeg.fif'
                session_file_paths.append((sub, fif_file))  # Store subject name and file path as tuple
            file_paths.append(session_file_paths)  # Store file paths for the current combination

# Loop through each file and update channel names with subject name
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
