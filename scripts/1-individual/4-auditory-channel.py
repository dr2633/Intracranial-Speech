import os
import numpy as np
import mne
from scipy.io.wavfile import write

# Params
freq = 'raw'
tasks = ('listen', 'speak')

# User paths
user_paths = [
    '/Users/derekrosenzweig/Documents/GitHub/iEEG-Speech',
    '/Users/lauragwilliams/Documents/projects/iEEG-Speech'
]

# Determine the base path
base_path = next((path for path in user_paths if os.path.exists(path)), None)
if base_path is None:
    raise Exception("Base path not found. Please check the user paths.")

# Set path to BIDS folder
BIDS_path = os.path.join(base_path, 'BIDS')

# List all files to loop through
session_stimuli = {
    'ses-01': ['Jobs1', 'Jobs2', 'Jobs3'],
    'ses-02': ['AttFast', 'AttSlow', 'BecFast', 'BecSlow', 'CampFast', 'CampSlow']
}

runs = ['run-01', 'run-02']
subjects = ['sub-01','sub-02']

for sub in subjects:
    for ses, stims in session_stimuli.items():
        for stim in stims:
            for run in runs:
                for task in tasks:
                    # Define the directory path for saving
                    save_directory = os.path.join(base_path, 'derivatives', 'auditory_channel', sub, ses, task)
                    if not os.path.exists(save_directory):
                        os.makedirs(save_directory)

                    # Get auditory channel from the raw edf file
                    edf_path = os.path.join(BIDS_path, sub, ses, 'ieeg', f'{sub}_{ses}_task-{task}{stim}_{run}_ieeg.edf')

                    # Skip processing if EDF file does not exist
                    if not os.path.exists(edf_path):
                        print(f"File does not exist: {edf_path}, skipping...")
                        continue

                    data = mne.io.read_raw_edf(edf_path, preload=True)


                    # # Load edf data -- resample to 1000 Hz if not already
                    # print(f"Processing file: {edf_path}")
                    # data = mne.io.read_raw_edf(edf_path, preload=True)
                    # if data.info['sfreq'] != 1000:
                    #     data.resample(1000)


                    # Get the auditory channel (assuming it's always the second channel, index 1)
                    auditory = np.copy(data._data[1, :])
                    print(f"Shape of auditory data: {np.shape(auditory)}")

                    # Set the WAV file path
                    auditory_wav_file_path = os.path.join(save_directory, f'{sub}_{ses}_{task}{stim}_{run}_auditory.wav')

                    # Convert the numpy array to a WAV file
                    rate = 10000  # Sample rate
                    scaled = np.int16(auditory / np.max(np.abs(auditory)) * 32767)  # Expected range for a WAV file
                    write(auditory_wav_file_path, rate, scaled)

                    print(f'Auditory data for task "{task}" saved as WAV: {auditory_wav_file_path}')
