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

#### Preprocessing 

The raw sEEG recordings are preprocessed using a pipeline implemented leveraging the MNE-Python library.  

#### Visualizing Stimulus Onset Time


To reproduce the visualization of the stimulus onset recorded in the microphone channel, run [stimulus_onset_time.py](stimulus_onset_time.py).

Below is a figure showing the stimulus onset time extracted from a representative session:

<p align="middle">
  <img src="figures/sub-01_ses-01_Jobs2_run-01_onset.jpg" width="70%" />
</p>


#### Word and Phoneme Evoked Response


To reproduce the visualization of evoked response to word and phoneme onset, run [preprocessing-filtered-data.py](preprocessing-filtered-data.py).

Below are figures showing evoked response to word onsets (averaged response across all words) and evoked response to phoneme onsets (averaged response across all phonemes) in an individual participant over the course of a single trial.


<p align="middle">
  <img align="top" src="figures/70-150Hz/word-evoked-sub-03-ses-02-BecSlow-run-01.jpg" width="45%" />
  <img align="top" src="figures/70-150Hz/phoneme-evoked-sub-03-ses-02-AttFast-run-01.jpg" width="45%" />
</p>

**** 

### Annotations 

**Audio Features**



<p align="middle">
  <img align="top" src="figures/F0-Spectrogram.png" width="45%" />
  <img align="top" src="figures/Intensity-Waveform.png" width="45%" />
</p>


**Language Features**

<p align="middle">
  <img align="top" src="figures/Jobs1_embeddings_sentence.png" width="45%" />
  <img align="top" src="figures/Jobs1_entropy_sentence.png" width="45%" />
</p>

**Features in CCA with Electrode Recordings**

<p align="middle">
  <img src="figures/Jobs1-feature-plot.jpg" width="70%" />
</p>





