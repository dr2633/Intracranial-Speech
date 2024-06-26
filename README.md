## sEEG Speech Decoding Project 

This repository contains code and data for preprocessing and analyzing stereotactic electroencephalography (sEEG) data aligned with audio and language features in a speech task. 

### Prerequisites

Before setting up the project, ensure you have Conda installed on your system. If not, you can install it from [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution).

### Setup and Activate Environment 

The project uses a Conda environment to manage dependencies. The environment configuration is provided in a file named `mne_flow.yml`, which includes all necessary packages such as `mne`, `torch`, and `openai-whisper`.

#### Key Dependencies
- `mne`: For EEG data analysis.
- `torch`, `torchaudio`: For processing audio and electrode data. 
- `pysurfer`: For visualization of neuroimaging data.
- `openai-whisper`: For state-of-the-art speech recognition -- access to whisper for transcription. 
- `praat-parselmouth`: For integrating the Praat software into Python for voice analysis.

#### Environment Setup

1. **Update Conda** (optional, but recommended):
   ```bash
   conda update -n base -c defaults conda


#### Stimuli 

Our study used a set of speech stimuli, presented in two separate sessions. Speeches were divided into approximately five-minute segments, with some segments played at the original speed and others at an accelerated rate (1.25x speed). 

In this repository, we provide a single stimuli used in the experiment. An excerpt from a Steve Jobs speech as a wav file along with text transcription derived from Whisper. 

****

A 2-second sine wave was played at the beginning of each speech segment to facilitate alignment of neural responses to stimulus onset. The tone onset recorded in the microphone channel was correlated with the corresponding wav file used to identify and store the precise stimulus onset time for each presentation. This method for correlating the wav file with the microphone channel recording is particularly useful in neuroscience research using ecologically valid and complex speech stimuli 

#### Annotations 

**Audio Features**


**Language Features**


#### Preprocessing 

**Stimulus Onset**

**Word Evoked Response**

**Phoneme Evoked Response**





<p align="middle">
  <img align="top" src="results/real_data_ablations/Student-Teacher/unablated.png" width="45%" />
  <img align="top" src="results/real_data_ablations/California%20Housing/unablated.png" width="45%" />
  <img align="top" src="results/real_data_ablations/Diabetes/unablated.png" width="45%" />
  <img align="top" src="results/real_data_ablations/WHO%20Life%20Expectancy/unablated.png" width="45%" />
</p>