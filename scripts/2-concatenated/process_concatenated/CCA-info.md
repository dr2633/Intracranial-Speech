# iEEG-Audio Canonical Correlation Analysis

This project applies Canonical Correlation Analysis (CCA) to intracranial electroencephalography (iEEG) data and audio information, specifically for identifying electrodes of interest in decoding speech from electrode activity.

## Background

Intracranial electroencephalography (iEEG) is a technique for recording electrical activity directly from the human brain, providing high spatial and temporal resolution. In the context of speech processing, iEEG data provides insight into the neural mechanisms underlying speech processing.

Canonical Correlation Analysis (CCA) is a statistical method that explores the relationship between two multivariate sets of variables. By applying CCA to iEEG data (electrodes) and audio (Mel coefficients), we identify linear combinations of electrode activity and audio audio features that maximize their correlation. This analysis helps uncover electrodes most informative in decoding audio from neural activity.

## Relevance

The application of CCA to iEEG data and audio information has several important implications:

1. **Speech Decoding**: By identifying the electrodes that exhibit strong correlations with audio features, CCA can aid in decoding speech from neural activity. This has potential applications in brain-computer interfaces, speech restoration for individuals with communication disorders, and understanding the neural basis of speech processing.

2. **Electrode Selection**: CCA can help identify the most informative electrodes for speech decoding tasks. By focusing on the electrodes that exhibit high canonical correlations, researchers can prioritize their analysis and reduce the dimensionality of the iEEG data, leading to more efficient and targeted investigations.

3. **Neuroscientific Insights**: CCA results can provide valuable insights into the anatomical localization and spatial dynamics of neural populations involved in speech processing. 

## Project Goals

The main goals of this project are:

1. Implement CCA between iEEG data and audio information (F0 and Mel coefficients) 
2. Identify electrodes that exhibit strong correlations with audio features.
3. Visualize and interpret the results of the CCA analysis.
4. Develop a Python package that encapsulates the CCA analysis pipeline for easy reuse and sharing.

By achieving these goals, this project aims to contribute to the field of speech processing and neuroscience by providing a valuable tool for analyzing iEEG data and uncovering the neural correlates of speech.

## Getting Started

To get started with this project, please refer to the installation instructions, usage guidelines, and API documentation provided in the repository.


## Papers 

https://openreview.net/pdf?id=X3TdREzbZN

https://openreview.net/pdf?id=W3cDd5xlKZ

https://arxiv.org/pdf/1812.02598.pdf

https://hal.science/hal-02541124/document

https://openreview.net/group?id=ICLR.cc/2024/Conference#tab-accept-oral

https://openreview.net/pdf?id=oO6FsMyDBt










