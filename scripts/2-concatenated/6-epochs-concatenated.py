import mne
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

# Set parameters for the concatenated file path
freq = '1-40Hz'
pro = 'preprocessed'

# List all sessions, stimuli, and runs to loop through
session_stimuli = {
    'ses-01': ['Jobs1', 'Jobs2', 'Jobs3'],
    'ses-02': ['AttFast', 'AttSlow', 'BecFast', 'BecSlow', 'CampFast', 'CampSlow']
}
runs = ['run-01', 'run-02']

# Set base path depending on who is running the code
user_paths = ['/Users/derekrosenzweig/Documents/GitHub/iEEG-Speech',
              '/Users/lauragwilliams/Documents/projects/iEEG-Speech']

for user_path in user_paths:
    if os.path.exists(user_path):
        base_path = user_path

BIDS_path = os.path.join(base_path, 'BIDS')

for stim in stimuli:

        for run in runs:
            # Construct file path to concatenated files
            fif_path = f'{base_path}/derivatives/concatenated/{freq}/{pro}/{stim}_{run}_concatenated_raw.fif'

            word_path = os.path.join(base_path, 'annotations', 'words', 'tsv', f'{stim}-words.tsv')
            phoneme_path = os.path.join(base_path, 'annotations', 'phonemes', 'tsv', f'{stim}-phonemes.tsv')

            # Check if file exists
            if not os.path.exists(fif_path):
                print(f"File does not exist: {fif_path}, skipping...")
                continue

            print(f"Processing file: {fif_path}")

            # Read raw fif file
            raw_CAR = mne.io.read_raw_fif(fif_path, preload=True)

            # Get data from MNE object
            CAR_z = raw_CAR.get_data()

            # Load metadata for phonemes from annotations
            phoneme_info = pd.read_csv(phoneme_path, delimiter='\t')

            # Create events for phoneme epochs
            phoneme_onsets = (phoneme_info['Start'].values * 1000.).astype(int)
            phoneme_events = np.column_stack(
                (phoneme_onsets, np.zeros_like(phoneme_onsets), np.ones_like(phoneme_onsets)))

            # Ensure the number of events matches the number of metadata rows
            min_len = min(len(phoneme_events), len(phoneme_info))
            phoneme_events = phoneme_events[:min_len]
            phoneme_info = phoneme_info[:min_len]

            # Print the lengths of events and metadata
            print("Length of events:", len(phoneme_events))
            print("Length of metadata:", len(phoneme_info))

            # Check if any duplicate indices are present in the metadata
            duplicate_indices = phoneme_info.index.duplicated(keep=False)
            if any(duplicate_indices):
                print("Duplicate indices found in metadata:", phoneme_info[duplicate_indices])

            # Check if any duplicate indices are present in the events
            unique_event_indices = np.unique(phoneme_events[:, 0], return_counts=True)
            if any(unique_event_indices[1] > 1):
                print("Duplicate indices found in events:", unique_event_indices[0][unique_event_indices[1] > 1])

            # Assign metadata to phoneme epochs
            phoneme_epochs = mne.Epochs(raw_CAR, phoneme_events, tmin=-0.2, tmax=0.6, baseline=None, reject=None,
                                        flat=None,
                                        event_repeated='drop')

            phoneme_epochs.metadata = phoneme_info

            # Define the directory path for saving phoneme epochs
            phoneme_epochs_dir = os.path.join(base_path, 'derivatives', 'concatenated', freq, 'phoneme_epochs')
            os.makedirs(phoneme_epochs_dir, exist_ok=True)

            # Save phoneme epochs
            phoneme_epochs_filename = f'phoneme-epo-{stim}-{run}-epo.fif'
            phoneme_epochs_filepath = os.path.join(phoneme_epochs_dir, phoneme_epochs_filename)
            phoneme_epochs.save(phoneme_epochs_filepath, overwrite=True)

            print("Phoneme epochs saved to:", phoneme_epochs_filepath)

            # Reload the saved phoneme epochs to verify metadata is saved
            loaded_phoneme_epochs = mne.read_epochs(phoneme_epochs_filepath, preload=True)

            if loaded_phoneme_epochs.metadata is not None:
                print("Metadata has been successfully saved and loaded with the epochs.")
                # Optionally print the first few rows to verify content
                print(loaded_phoneme_epochs.metadata.head())
            else:
                print("No metadata found in the loaded epochs.")

            # Load metadata for words from annotations
            word_info = pd.read_csv(word_path, delimiter='\t', encoding='utf-8')

            # Create events for words epochs
            word_onsets = (word_info['Start'].values * 1000)
            onset_samples_word = (np.array(word_onsets) / 1000 * raw_CAR.info['sfreq']).astype(int)
            word_events = np.column_stack(
                (onset_samples_word, np.zeros_like(onset_samples_word), np.ones_like(onset_samples_word)))

            # Create epochs with the filtered metadata for words
            word_epochs = mne.Epochs(raw_CAR, word_events, tmin=-.6, tmax=1.8, baseline=None, reject=None, flat=None)
            word_epochs.metadata = word_info  # Assign metadata to phoneme epochs

            # Define the directory path for saving word epochs
            word_epochs_dir = os.path.join(base_path, 'derivatives', 'concatenated', freq, 'word_epochs')
            os.makedirs(word_epochs_dir, exist_ok=True)

            # Save word epochs with a specific filename
            word_epochs_filename = f'word-epo-{stim}-{run}-epo.fif'
            word_epochs_filepath = os.path.join(word_epochs_dir, word_epochs_filename)

            # Save word epochs
            word_epochs.save(word_epochs_filepath, overwrite=True)
            print(f"Word epochs saved to {word_epochs_filepath}")

            # Print the lengths of events and metadata
            print("Length of events:", len(word_events))
            print("Length of metadata:", len(word_info))

            # Reload epochs to check that metadata is properly saved
            loaded_word_epochs = mne.read_epochs(word_epochs_filepath, preload=True)

            if loaded_word_epochs.metadata is not None:
                print("Metadata has been successfully saved and loaded with the epochs.")
                print(loaded_word_epochs.metadata.head())
            else:
                print("No metadata found in the loaded epochs.")

            # Average epochs for evoked response
            word_evoked = word_epochs.average()

            # Define the full file path for saving the evoked figure
            evoked_fig_name = f'word-evoked-{stim}-{run}.jpg'
            evoked_fig_path = os.path.join(base_path, 'vis', 'concatenated', freq, 'word_evoked', evoked_fig_name)
            os.makedirs(os.path.dirname(evoked_fig_path), exist_ok=True)

            # Plot and save evoked response
            fig, ax = plt.subplots(figsize=(15, 5))
            ax.plot(word_evoked.times, word_evoked.data.T)
            plt.savefig(evoked_fig_path, format='jpg', dpi=300)
            plt.close()

            # Average epochs for evoked response
            phoneme_evoked = phoneme_epochs.average()

            # Define the full file path for saving the evoked figure
            evoked_fig_name = f'phoneme-evoked-{stim}-{run}.jpg'
            evoked_fig_path = os.path.join(base_path, 'vis', 'concatenated', freq, 'phoneme_evoked', evoked_fig_name)
            os.makedirs(os.path.dirname(evoked_fig_path), exist_ok=True)

            # Plot and save evoked response
            fig, ax = plt.subplots(figsize=(15, 5))
            ax.plot(phoneme_evoked.times, phoneme_evoked.data.T)
            plt.savefig(evoked_fig_path, format='jpg', dpi=300)
            plt.close()

            # Save trial information as TSV
            trial_info_path = os.path.join(base_path, 'derivatives', 'concatenated', freq, 'trial_information')
            os.makedirs(trial_info_path, exist_ok=True)
            trial_info_file = os.path.join(trial_info_path, f'{stim}_{run}_trial_info.tsv')

            # Create a DataFrame with trial information
            trial_info = pd.DataFrame({
                'Stimulus': [stim],
                'Run': [run],
                'Frequency': [freq],
                'Preprocessed Data': [str(np.shape(CAR_z))],
                'Word Epochs': [str(np.shape(word_epochs))],
                'Phoneme Epochs': [str(np.shape(phoneme_epochs))]

            }, columns=['Subject', 'Session', 'Stimulus', 'Run', 'Frequency', 'Preprocessed Data', 'Word Epochs',
                        'Phoneme Epochs'])

            # Save the trial information as a TSV file
            trial_info.to_csv(trial_info_file, sep='\t', index=False)
            print(f"Trial information saved to {trial_info_file}")
