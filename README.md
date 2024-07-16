## Stereotactic Electroencephalography (sEEG) Speech Project

This repository contains code and data for preprocessing and analyzing stereotactic electroencephalography (sEEG) data aligned with audio and language features in a speech task. Our study used a set of speech stimuli, presented in two separate sessions. Speeches were divided into approximately five-minute segments, with some segments played at the original speed and others at an accelerated rate (1.25x speed). 

Intracranial data, time-resolved annotations, and stimuli are stored in Box and organized in accordance with Brain Imaging Data Structure (BIDS) format. 



### Prerequisites

Before setting up the project, ensure you have Conda installed on your system. If not, you can install it from [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution).

### Setup and Activate Environment 

The project uses a Conda environment to manage dependencies. The environment configuration is provided in a file named `mne_flow.yml`, which includes all necessary packages such as `mne`, `torch`, and `openai-whisper`.

#### Key Dependencies
- `mne`: For EEG data analysis.
- `torch`, `torchaudio`: For processing audio and electrode data. 
- `pysurfer`: For visualization of neuroimaging data.
- `openai-whisper`: For text transcription of `.wav` files. 
- `praat-parselmouth`: For integrating the Praat software into Python for voice analysis.

#### Environment Setup

1. **Update Conda** (optional, but recommended):
   ```bash
   conda update -n base -c defaults conda

<<<<<<< HEAD
### Preprocessing 

The raw sEEG recordings are preprocessed using a pipeline implemented leveraging the MNE-Python library. Here are the initial key steps: 

=======
#### Preprocessing 

The raw sEEG recordings are preprocessed using a pipeline implemented leveraging the MNE-Python library.  
>>>>>>> ca24bab4e496663c0e24b2de128837963ee1dd57

#### Visualizing Stimulus Onset Time

To reproduce the visualization of the stimulus onset recorded in the microphone channel, run [stimulus-onset.py](stimulus-onset.py).

Below is a figure showing the stimulus onset time extracted from a representative session:

<p align="middle">
  <img src="figures/sub-01_ses-01_Jobs2_run-01_onset.jpg" width="70%" />
</p>

<<<<<<< HEAD
A 2-second sine wave was played at the beginning of each speech segment to facilitate alignment of neural responses to stimulus onset. The tone onset recorded in the microphone channel was correlated with the corresponding wav file used to identify and store the precise stimulus onset time for each presentation. This method for correlating the wav file with the microphone channel recording is particularly useful in neuroscience research using ecologically valid stimuli.  


****

#### Word and Phoneme Evoked Response

To visualize evoked response to word and phoneme onsets in the stimulus, run [preprocessing-filtered-data.py](preprocessing-filtered-data.py)

Below is a figure showing the work evoked response (left) and phoneme evoked response (right) for a representative participant in a single trial. 
=======

#### Word and Phoneme Evoked Response


To reproduce the visualization of evoked response to word and phoneme onset, run [preprocessing-filtered-data.py](preprocessing-filtered-data.py).

Below are figures showing evoked response to word onsets (averaged response across all words) and evoked response to phoneme onsets (averaged response across all phonemes) in an individual participant over the course of a single trial.

>>>>>>> ca24bab4e496663c0e24b2de128837963ee1dd57

<p align="middle">
  <img align="top" src="figures/70-150Hz/word-evoked-sub-03-ses-02-BecSlow-run-01.jpg" width="45%" />
  <img align="top" src="figures/70-150Hz/phoneme-evoked-sub-03-ses-02-AttFast-run-01.jpg" width="45%" />
</p>

<<<<<<< HEAD
****

#### Annotations 
=======
**** 

### Annotations 
>>>>>>> ca24bab4e496663c0e24b2de128837963ee1dd57

**Audio Features**

We extract two features from the raw audio: fundamental frequency (Hz) and sound intensity (dB).


<p align="middle">
  <img align="top" src="figures/F0-Spectrogram.png" width="45%" />
  <img align="top" src="figures/Intensity-Waveform.png" width="45%" />
</p>

We use the audio features to identify and localize electrodes maximally responsive in auditory processing of the stimulus. 

**Language Features**

<<<<<<< HEAD
We extract language features from the text transcription of the speeches: GPT-2 Embeddings (5 Principal Components of eighth layer hidden activations) and GPT-2 Entropy (word-level entropy). 

=======
>>>>>>> ca24bab4e496663c0e24b2de128837963ee1dd57
<p align="middle">
  <img align="top" src="figures/Jobs1_embeddings_sentence.png" width="45%" />
  <img align="top" src="figures/Jobs1_entropy_sentence.png" width="45%" />
</p>

We use language features to identify electrodes maximally responsive to language processing in the recording sessions. 

**Features in CCA with Electrode Recordings**

<p align="middle">
  <img src="figures/Jobs1-feature-plot.jpg" width="70%" />
</p>



1. **Resampling the Electrode Data**: The electrode data is initially resampled from 10,000 Hz to 1,000 Hz. 
2. **Audio Extraction and Transcription**: The microphone channel from the raw data is saved, coverted to a `.wav` file and then transcribed using Whisper to capture both listening and speaking sessions.
3. **Stimulus Onset Time Extraction**: We extract the stimulus onset time by correlating the stimulus with the audio recorded in the channel. We save the the stimulus onset time in an events `.tsv` file in accordance with BIDS. 
4. **Visual Inspection**: For each trial, we visually inspect that the stimulus onset time extracted from the microphone channel corresponds with the raw audio used in stimulus presentation.    





