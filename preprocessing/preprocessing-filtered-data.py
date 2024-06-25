# Script for preprocessing edf files, creating epochs, saving derivatives and figures
import mne
import numpy as np
from process_nwb.linenoise_notch import apply_linenoise_notch
import pandas as pd
from process_nwb.common_referencing import subtract_CAR
from scipy.stats import zscore
from process_nwb.wavelet_transform import wavelet_transform, get_filterbank
import os
import matplotlib.pyplot as plt
import librosa
import pandas as pd


# Set parameters for EDF file
freq = '1-40Hz'

# Setting user paths
user_paths = [
    '/Users/derekrosenzweig/Documents/GitHub/iEEG-Speech',
    '/Users/lauragwilliams/Documents/projects/iEEG-Speech'
]

# Determine the base path
base_path = next((path for path in user_paths if os.path.exists(path)), None)

# List of files to loop through
session_stimuli = {
    'ses-01': ['Jobs1', 'Jobs2', 'Jobs3'],
    'ses-02': ['AttFast', 'AttSlow', 'BecFast', 'BecSlow', 'CampFast', 'CampSlow']
}

# sub-08 Jobs1 run-01 and CampSlow


runs = ['run-01','run-02']

subjects  = ['sub-08']


for sub in subjects:
    for ses, stims in session_stimuli.items():
        for stim in stims:
            for run in runs:
                print(f"Processing {stim} in {ses}, {run}...")

                # Paths based on current session and stimulus
                BIDS_path = os.path.join(base_path, 'BIDS')
                edf_path = os.path.join(BIDS_path, sub, ses, 'ieeg', f'{sub}_{ses}_task-listen{stim}_{run}_ieeg.edf')
                onset_path = os.path.join(BIDS_path, sub, ses, 'ieeg', f'{sub}_{ses}_task-listen{stim}_{run}_events.tsv')
                fif_path = os.path.join(base_path, 'derivatives', 'individual', freq, 'preprocessed',
                                        f'{sub}_{ses}_task-listen{stim}_{run}_ieeg.fif')
                word_path = os.path.join(base_path, 'annotations', 'words', 'tsv', f'{stim}-words.tsv')
                phoneme_path = os.path.join(base_path, 'annotations', 'phonemes', 'tsv', f'{stim}-phonemes.tsv')
                stim_path = os.path.join(base_path, 'stimuli', 'wav', f'{stim}.wav')

                # Setting electrode path for saving channel information
                elec_path = os.path.join(BIDS_path, sub,ses,'ieeg', f'{sub}_{ses}_electrodes.tsv')

                # Read electrode information from the TSV file
                electrode_info = pd.read_csv(elec_path, sep='\t')

                # Select columns to add for channel information
                channel_cols = ['name', 'x', 'y', 'z', 'WMvsGM', 'LvsR', 'Desikan_Killiany']

                # Extract the selected columns from the electrode information
                channel_metadata = electrode_info[channel_cols]

                # Skip processing if EDF file does not exist
                if not os.path.exists(edf_path):
                    print(f"File does not exist: {edf_path}, skipping...")
                    continue

                # Read in raw data and resample to 1000 Hz
                raw = mne.io.read_raw_edf(edf_path, preload=True)
                if raw.info['sfreq'] != 1000:
                    raw.resample(1000)

                # Get stimulus onset time from events TSV
                df = pd.read_csv(onset_path, sep='\t')
                onset_time = df['onset'].tolist()
                stim_onset = onset_time[0] * 1000  # Convert to milliseconds
                stim_onset_sec = stim_onset / 1000.0  # Convert milliseconds to seconds

                # Conditional -- if stimulus onset begins before recording
                if stim_onset_sec < 0:
                    tmin = 0
                else:
                    tmin = stim_onset_sec

                # Get duration of WAV file in ms
                y, sr = librosa.load(stim_path, sr=None)
                wav_duration = librosa.get_duration(y=y, sr=sr)

                # Get the last timepoint in the raw data
                raw_duration_sec = raw.times[-1]

                # Clip the raw data to match the duration of the WAV file
                if tmin + wav_duration > raw_duration_sec:
                    clipped_raw = raw.copy().crop(tmin=tmin, tmax=raw_duration_sec)
                else:
                    clipped_raw = raw.copy().crop(tmin=tmin, tmax=tmin + wav_duration)

                print(np.shape(clipped_raw._data))

                # Apply notch filter to remove 60 Hz line noise and harmonics
                nth_data = apply_linenoise_notch(clipped_raw._data[5:, :].T, clipped_raw.info['sfreq']).T

                # Check if stimulus onset is negative
                if stim_onset_sec < 0:
                    # Calculate number of samples to pad
                    num_samples_to_pad = int(abs(stim_onset_sec) * raw.info['sfreq'])

                    # Create zero padding
                    padding = np.zeros((len(nth_data), num_samples_to_pad))

                    # Apply zero padding to preprocessed data
                    nth_data = np.hstack((padding, nth_data))

                # Get the correct index of channels in the iEEG data (microphone channels removed)
                num_channels = nth_data.shape[0]
                max_channel_index = min(len(raw.info['ch_names']), 5 + num_channels)
                new_channel_indices = range(5, max_channel_index)
                updated_info = mne.pick_info(raw.info, sel=new_channel_indices)
                print(np.shape(nth_data))

                # Create MNE RawArray object with notch filtered data and updated index
                nth_raw = mne.io.RawArray(nth_data, updated_info)

                # Apply CAR and Z-score
                nth_CAR = subtract_CAR(nth_raw.get_data())
                raw_CAR = mne.io.RawArray(nth_CAR, nth_raw.info)
                print(raw_CAR.info)

                if freq == '1-15Hz':
                    raw_CAR.filter(1, 15)
                    CAR_z = zscore(raw_CAR._data, axis=1)

                if freq == '1-40Hz':
                    raw_CAR.filter(1, 40)
                    CAR_z = zscore(raw_CAR._data, axis=1)

                elif freq == '70-150Hz':
                    # compute
                    nth_CAR_transpose = raw_CAR._data.T
                    tf_data, _, ctr_freq, bw = wavelet_transform(nth_CAR_transpose, raw.info['sfreq'],
                                                                 filters='changlab', hg_only=True)

                    # Z scoring the amplitude instead of the complex waveform
                    tf_data = abs(tf_data)

                    # average over time, to get a mean value for each channel and freq band
                    mean = tf_data.mean(axis=0, keepdims=True)

                    # std over time, to get a stf for each channel and freq band
                    std = tf_data.std(axis=0, keepdims=True)

                    # subtract the mean of each channel and freq, divide by std to get zscore
                    tf_norm_data = (tf_data - mean) / std

                    # average over the z-scored freq bands
                    high_gamma_transpose = tf_norm_data.mean(axis=-1)

                    # put back into the original shape
                    high_gamma = high_gamma_transpose.T

                    # zscore again, on the average over freq bands
                    CAR_z = zscore(high_gamma, axis=0)

                    # Return the center frequencies and bandwidths used in the wavelet transform
                    print(f'Shape of bandpass filtered analytic signal {np.shape(tf_data)}')
                    print(f'Center frequencies used:{ctr_freq}')
                    print(f'Bandwidths used:{bw}')
                    print(f'Shape of filtered and z-scored signal: {np.shape(CAR_z)}')

                # put z scored and filtered data into the raw mne object
                raw_CAR._data = np.array(CAR_z, dtype='float64')  # needs to be float64 to save epochs

                # Save the raw_CAR object as a fif file
                os.makedirs(os.path.dirname(fif_path), exist_ok=True)
                raw_CAR.save(fif_path, overwrite=True)
                print(f"Data saved to {fif_path}")

                # Load metadata for phonemes from annotations
                phoneme_info = pd.read_csv(phoneme_path, delimiter='\t')

                # Create events for phoneme epochs
                phoneme_onsets = (phoneme_info['Start'].values * 1000.).astype(int)
                phoneme_events = np.column_stack((phoneme_onsets, np.zeros_like(phoneme_onsets), np.ones_like(phoneme_onsets)))

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
                phoneme_epochs = mne.Epochs(raw_CAR, phoneme_events, tmin=-0.2, tmax=0.6, baseline=None, reject=None, flat=None,
                                            event_repeated='drop')

                phoneme_epochs.metadata = phoneme_info

                # Define the directory path for saving phoneme epochs
                phoneme_epochs_dir = os.path.join(base_path, 'derivatives', 'individual', freq, 'phoneme_epochs')
                os.makedirs(phoneme_epochs_dir, exist_ok=True)

                # Save phoneme epochs
                phoneme_epochs_filename = f'phoneme-epo-{sub}-{ses}-{stim}-{run}-epo.fif'
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
                word_epochs_dir = os.path.join(base_path, 'derivatives', 'individual', freq, 'word_epochs')
                os.makedirs(word_epochs_dir, exist_ok=True)

                # Save word epochs with a specific filename
                word_epochs_filename = f'word-epo-{sub}-{ses}-{stim}-{run}-epo.fif'
                word_epochs_filepath = os.path.join(word_epochs_dir, word_epochs_filename)

                # Save word epochs
                word_epochs.save(word_epochs_filepath, overwrite=True)
                print(f"Word epochs saved to {word_epochs_filepath}")

                # Print the lengths of events and metadata
                print("Length of events:", len(word_events))
                print("Length of metadata:", len(word_info))

                #Reload epochs to check that metadata is properly saved
                loaded_word_epochs = mne.read_epochs(word_epochs_filepath, preload=True)

                if loaded_word_epochs.metadata is not None:
                    print("Metadata has been successfully saved and loaded with the epochs.")
                    print(loaded_word_epochs.metadata.head())
                else:
                    print("No metadata found in the loaded epochs.")

                # Average epochs for evoked response
                word_evoked = word_epochs.average()

                # Define the full file path for saving the evoked figure
                evoked_fig_name = f'word-evoked-{sub}-{ses}-{stim}-{run}.jpg'
                evoked_fig_path = os.path.join(base_path, 'vis', 'individual', 'evoked', freq, 'word_evoked', evoked_fig_name)
                os.makedirs(os.path.dirname(evoked_fig_path), exist_ok=True)



                # Save trial information as TSV
                trial_info_path = os.path.join(base_path, 'derivatives', 'individual',freq, 'trial_information')
                os.makedirs(trial_info_path, exist_ok=True)
                trial_info_file = os.path.join(trial_info_path, f'{sub}_{ses}_task-listen{stim}_{run}_trial_info.tsv')

                # Create a DataFrame with trial information
                trial_info = pd.DataFrame({
                    'Subject': [sub],
                    'Session': [ses],
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