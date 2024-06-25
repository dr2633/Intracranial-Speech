# Stores functions used in scripts
import mne
import numpy as np
import pandas as pd
from scipy.io.wavfile import read
from librosa import resample
from scipy.signal import stft
import librosa
import csv
import os
import re


# Get clipping function for fif

# Function to save tsv
def save_as_tsv(tsv_path, onset, duration, sub, ses, stim, run):
    with open(tsv_path, 'w', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(['onset', 'duration', 'participant', 'session', 'stimuli', 'run'])
        writer.writerow([onset, duration, sub, ses, stim, run])

# for plotting
def elecs_to_mne(MNI_matrix):

    # make sure that the matrix is the right way around
    n_elecs, n_dims = MNI_matrix.shape
    if n_dims > n_elecs:
        MNI_matrix = MNI_matrix.T
        n_elecs, n_dims = MNI_matrix.shape

    # extract elec positions
    ch_names = [str(i) for i in range(n_elecs)]

    # make mne montage
    montage = mne.channels.make_dig_montage(ch_pos=dict(zip(ch_names, MNI_matrix)),
                                            coord_frame='fs_tal')

    # and info
    info = mne.create_info(ch_names, 1000., 'seeg').set_montage(montage)
    return info, montage


def stimulus_onset(sub, ses, stim, run, user_paths, target_freq=440, target_sr=1000):
    """
    Detect the onset of a tone in an auditory EEG channel and
    calculate the onset time of the auditory channel from the EDF file.

    Parameters:
    sub: str - subject identifier
    ses: str - session identifier
    stim: str - stimulus identifier
    run: str - run identifier
    user_paths: list - list of potential paths to data directories
    target_freq: int - target frequency to detect (default 440 Hz)
    target_sr: int - target sample rate for resampling (default 1000 Hz)

    Returns:
    onset_time: float or None - time in seconds when the tone starts, or None if not detected
    """

    # Set base path depending on who is running the code
    for user_path in user_paths:
        if os.path.exists(user_path):
            base_path = user_path
            break
    else:
        raise FileNotFoundError("No valid path found in user_paths.")

    edf_path = f'{base_path}/{sub}/{ses}/ieeg/{sub}_{ses}_task-listen{stim}_{run}_ieeg.edf'
    wav_file = f'{base_path}/stimuli/{stim}.wav'

    # Check if the files exist
    if not os.path.exists(edf_path) or not os.path.exists(wav_file):
        raise FileNotFoundError("EDF or WAV file not found.")

    # Load the EDF data
    data = mne.io.read_raw_edf(edf_path, preload=True)
    auditory = np.copy(data._data[1, :])  # Assuming the second channel is the auditory channel
    seeg_sfreq = np.copy(int(data.info['sfreq']))

    # Resample to the target sample rate
    resampled_seeg = resample(y=auditory, orig_sr=seeg_sfreq, target_sr=target_sr)

    # Perform STFT to find the onset time
    nperseg = int(target_sr / 2)
    noverlap = nperseg // 2
    frequencies, times, Zxx = stft(resampled_seeg, fs=target_sr, nperseg=nperseg, noverlap=noverlap)
    freq_idx = np.argmin(np.abs(frequencies - target_freq))
    power_at_target_freq = np.abs(Zxx[freq_idx, :])
    threshold = np.mean(power_at_target_freq) + 2 * np.std(power_at_target_freq)
    onset_indices = np.where(power_at_target_freq > threshold)[0]

    if onset_indices.size == 0:
        print("No onset detected.")
        return None
    else:
        onset_idx = onset_indices[0]
        onset_time = times[onset_idx]
        print(f"The tone starts at approximately {onset_time} seconds.")

        return onset_time


def load_raw_data(edf_path):
    return mne.io.read_raw_edf(edf_path, preload=True)

def create_bipolar_pairs(ch_names):
    prefix_groups = {}
    for ch in ch_names:
        match = re.match(r"(POL [A-Za-z]+)\d+-Ref", ch)  # Adjust regex as needed
        if match:
            prefix = match.group(1)
            if prefix not in prefix_groups:
                prefix_groups[prefix] = []
            prefix_groups[prefix].append(ch)

    bipolar_pairs = []
    for group_ch_names in prefix_groups.values():
        for i in range(len(group_ch_names) - 1):
            bipolar_pairs.append((group_ch_names[i], group_ch_names[i + 1]))

    return bipolar_pairs

def apply_bipolar_reference(raw_data, bipolar_pairs):
    return mne.set_bipolar_reference(raw_data, anode=[pair[0] for pair in bipolar_pairs],
                                     cathode=[pair[1] for pair in bipolar_pairs],
                                     copy=True)

def filter_data(raw_data, l_freq, h_freq):
    return raw_data.filter(l_freq, h_freq, fir_design='firwin')


#used for converting electrode positions in excel to TSV format for BIDS
def excel_to_tsv(input_excel_file, output_tsv_file):


    try:
        # Read the Excel file into a pandas DataFrame
        df = pd.read_excel(input_excel_file)

        # Save the DataFrame to a TSV file
        df.to_csv(output_tsv_file, sep='\t', index=False)

        print(f'Successfully converted {input_excel_file} to {output_tsv_file}')
    except Exception as e:
        print(f'Error: {str(e)}')

    excel_to_tsv(input_excel_file, output_tsv_file)



def create_lead_groups(ch_names):
    lead_groups = {}
    for ch in ch_names:
        # General regex pattern to match different naming conventions
        match = re.match(r"([A-Za-z]+\s*[A-Za-z]*)\d+-Ref", ch)
        if match:
            prefix = match.group(1)  # Extracts the prefix
            if prefix not in lead_groups:
                lead_groups[prefix] = []
            lead_groups[prefix].append(ch)
        else:
            print("No match for channel name:", ch)
    return lead_groups

def apply_leadcar_reference(raw_data, lead_groups):
    for lead, group_ch_names in lead_groups.items():
        available_ch_names = [ch for ch in group_ch_names if ch in raw_data.ch_names]
        if not available_ch_names:
            print(f"No channels from group '{lead}' are present in the raw data.")
            continue

        try:
            group_data = raw_data.copy().pick(available_ch_names).get_data()
            group_avg = np.mean(group_data, axis=0)
            for ch_name in available_ch_names:
                ch_index = raw_data.ch_names.index(ch_name)
                raw_data._data[ch_index] -= group_avg
        except ValueError as e:
            print(f"Error processing lead group '{lead}':", e)
    return raw_data


def get_wav_duration(wav_file):
    y, sr = librosa.load(wav_file, sr=None)  # Load the wav file
    duration = librosa.get_duration(y=y, sr=sr)
    return duration
