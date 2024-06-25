# Script for getting the stimulus onset time for each recording session

import os
import numpy as np
import mne

from scipy.signal import resample
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from scipy.io.wavfile import read


from preprocessing_functions import get_wav_duration, save_as_tsv

# Parameters
sub = "sub-08"
ses = "ses-01"
stim = 'Jobs1'
run = 'run-02'

# [10, 40, 0, 60]
# 'sub-02-Jobs3-listen-run-02': [195,210,180,220]
# 'sub-02-BecFast-listen-run-01': [5, 25, 0, 40]
# 'sub-05_ses-02_BecSlow_run-01
# 'sub-07-Jobs2-listen-run-01': [195,210,180,220]
# 'sub-07-Jobs3-listen-run-02': [1,15,0,20] --- listen to the audio


# Define additional parameters for stimulus onset time
stim_onset, stim_offset = 10,40
rec_onset, rec_offset = 0,60
decim, target_sr = 10, 1000

user_paths = ['/Users/derekrosenzweig/Documents/GitHub/iEEG-Speech']

# set base path depending on who is running the code
for user_path in user_paths:
    if os.path.exists(user_path):
        base_path = user_path

# Setting paths to files
edf_path = f'{base_path}/BIDS/{sub}/{ses}/ieeg/{sub}_{ses}_task-listen{stim}_{run}_ieeg.edf'
tsv_path = f'{base_path}/BIDS/{sub}/{ses}/ieeg/{sub}_{ses}_task-listen{stim}_{run}_events.tsv'
stim_path = f'{base_path}/stimuli/wav/{stim}.wav'
fig_path = f'{base_path}/vis/individual/onset/{sub}/{ses}/{stim}'

# Directory for saving figures for a specific subject
sub_figs_dir = os.path.join(fig_path, sub, ses, stim)
os.makedirs(sub_figs_dir, exist_ok=True)


# Load stim_onset_time function
def stim_onset_time(sub, stim, ses, run, edf_path, sub_figs_dir, stim_onset=0, stim_offset=30,
                    rec_onset=0, rec_offset=60, decim=10, plot=True, target_sr=1000):

    '''
    Get the timesample where the stimulus begins to play.

    subject: subject number, e.g., S23
    stimulus: wavefile name, e.g., 'Jobs1.wav'
    full path to edf data
    stim_onset: onset (in seconds) of stim window
    stim_offset: offset (in seconds) of stim window
    decim: how far apart to test onset (e.g. every 10 ms)
    target_sr: what sample rate to compare signals at

    returns:
    time of onset of wav file in seconds

    '''

    # load data
    data = mne.io.read_raw_edf(edf_path, preload=True)

    # get the auditory channel and sample freq
    auditory = np.copy(data._data[1, :])
    seeg_sfreq = np.copy(int(data.info['sfreq']))

    # clear memory
    del(data)

    # load the original wav file
    wav_file = f'{base_path}/stimuli/wav/{stim}.wav'

    wav_sfreq, wav = read(wav_file)
    wav = np.array(wav[:, 0], dtype='float')

    # resample to the same as the seeg recording
    num_samples_wav = int(len(wav) * target_sr / wav_sfreq)
    resampled_wav = resample(wav, num_samples_wav)

    num_samples_seeg = int(len(auditory) * target_sr / seeg_sfreq)
    resampled_seeg = resample(auditory, num_samples_seeg)

    # window of the original wav file
    wav_start = stim_onset*target_sr
    wav_stop = stim_offset*target_sr
    wav_size = int(wav_stop-wav_start)

    # take X seconds of the original wav, and then slide it across the recording
    wav_snippet = resampled_wav[wav_start:wav_stop]
    wav_snippet = wav_snippet / np.nanmean(wav_snippet)

    # window of the original wav file
    rec_start = rec_onset*target_sr
    rec_stop = rec_offset*target_sr
    rec_size = int(rec_stop-rec_start)

    # loop through every 10th millisecond to get the best correlation
    num_samples = int((rec_size/decim)+1)
    cor_vals = np.zeros(num_samples)
    sample_array = np.array(np.linspace(rec_start, rec_stop, num_samples), dtype='int')
    for ti, tt in enumerate(sample_array):
        test_signal = resampled_seeg[tt:wav_size+tt]
        test_signal = test_signal / np.nanmean(test_signal)
        r, _ = pearsonr(wav_snippet, test_signal)
        cor_vals[ti] = r

    # find the moment of max correlation
    tt_max_correlation = np.argmax(np.abs(cor_vals))*decim

    # now plot them on top of each other, with the padding added
    length_pad = len(wav_snippet)+tt_max_correlation
    padded_real = np.zeros(length_pad)
    padded_real[tt_max_correlation:] = wav_snippet



    # what was the best correlation it found?
    max_cor = np.max(np.abs(cor_vals))
    if max_cor < 0.1:
        print('Alignment was weak, please visually check before continuing.')

    # convert the timestamp into seconds
    tt_seconds = tt_max_correlation / np.float64(target_sr)

    # correct for the start point of the wav file
    rec_stim_diff = rec_onset-stim_onset
    tt_seconds = tt_seconds+rec_stim_diff

    if plot:

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.plot(cor_vals)
        plt.xlabel("Delays")
        plt.ylabel("Pearson Correlation")

        # Construct the figure file path using fig_dir
        fig_name = f'{sub}_{ses}_{stim}_{run}_delay.jpg'
        fig_file_path = os.path.join(sub_figs_dir, fig_name)

        # Save the figure
        plt.savefig(fig_file_path, format='jpg', dpi=300)
        plt.close()

        fig, ax = plt.subplots(1, 1, figsize=(20, 5))
        ax.plot(resampled_seeg[:len(padded_real)], label='Auditory Channel')
        ax.plot(padded_real * 10., label='WAV File')
        onset_time_x = tt_seconds * target_sr
        ax.axvline(x=onset_time_x, color='grey', linestyle='--', label=f'Onset Time: {tt_seconds:.2f} s')
        ax.set_ylim([-4e5, 4e5])

        # Display the onset time on the plot
        ax.text(onset_time_x + target_sr * 0.5, 0, f'Onset Time = {tt_seconds:.2f} s',
                verticalalignment='bottom', horizontalalignment='right',
                color='grey', fontsize=10)

        plt.legend()
        plt.title(f"Alignment between Auditory Channel and Wav File ({sub}_{ses}_{stim}_{run})")
        plt.xlabel("Time (ms)")
        plt.ylabel("Amplitude")

        # Save the figure
        # Construct the figure file path using fig_dir
        fig_name = f'{sub}_{ses}_{stim}_{run}_onset.jpg'
        fig_file_path = os.path.join(sub_figs_dir, fig_name)

        # Save the figure
        plt.savefig(fig_file_path, format='jpg', dpi=300)
        plt.close()

    return tt_seconds

# Call the function
onset_time = stim_onset_time(sub=sub, stim=stim, ses=ses, run=run, edf_path=edf_path,
                             sub_figs_dir=sub_figs_dir, stim_onset=stim_onset,
                             stim_offset=stim_offset, rec_onset=rec_onset,
                             rec_offset=rec_offset, decim=decim, plot=True, target_sr=target_sr)


# Get wav duration and save to events TSV for BIDS
wav_duration = get_wav_duration(stim_path)
save_as_tsv(tsv_path, onset_time, wav_duration, sub, ses, stim, run)
print(f"TSV saved to {save_as_tsv(tsv_path, onset_time, wav_duration, sub, ses, stim, run)}")