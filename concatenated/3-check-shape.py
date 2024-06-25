import mne

# Set parameters for path
stim = 'Jobs2'
ses = 'ses-01'
run = 'run-01'
freq = '70-150Hz'
pro = 'preprocessed'

base_path = f'/Users/derekrosenzweig/Documents/GitHub/iEEG-Speech/derivatives/individual/{freq}/{pro}'

# Create an empty list to store file paths
file_paths = []

# Create a for loop that gets files for sub = 'sub-01' through 'sub-07'
for i in range(1, 8):  # Iterate through sub-01 to sub-07
    sub = f'sub-{i:02d}'
    fif_file = f'{base_path}/{sub}_{ses}_task-listen{stim}_{run}_ieeg.fif'
    file_paths.append(fif_file)

# Create an empty list to store the raw data from each file
raw_data_list = []

# Read raw data from each file, rename channels, and append to the list
for file_path in file_paths:
    try:
        raw = mne.io.read_raw_fif(file_path)
        # Get the subject ID from the file path
        sub_id = file_path.split('/')[-1].split('_')[0]
        # Rename channels with subject ID
        for idx, channel in enumerate(raw.ch_names):
            raw.rename_channels({channel: f"{sub_id} {channel}"})
        # Append the raw data to the list
        raw_data_list.append(raw)
        print(f"Channel names in file '{file_path}' with subject '{sub_id}' updated.")
        print(raw.ch_names)
        # Print the shape of the raw data for the current file
        print(f"Shape of raw data for file '{file_path}': {raw.get_data().shape}")
    except FileNotFoundError:
        print(f"File '{file_path}' not found. Skipping...")
        continue

print("Shape checking complete.")
