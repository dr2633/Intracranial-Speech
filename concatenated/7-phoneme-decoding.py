import mne
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, KFold
import os
import numpy as np
import csv

# Set parameters for the concatenated file path
freq = '70-150Hz'
pro = 'phoneme_epochs'
features = ('phonation', 'manner', 'place')

# List all sessions, stimuli, and runs to loop through
stimuli = ['Jobs1', 'Jobs2', 'Jobs3', 'AttFast', 'AttSlow', 'BecFast', 'BecSlow', 'CampFast', 'CampSlow']
runs = ['run-01', 'run-02']

# Set base path depending on who is running the code
user_paths = ['/Users/derekrosenzweig/Documents/GitHub/iEEG-Speech',
              '/Users/lauragwilliams/Documents/projects/iEEG-Speech']

for user_path in user_paths:
    if os.path.exists(user_path):
        base_path = user_path
        break
else:
    raise FileNotFoundError("No valid base path found.")

# Create an empty dictionary to store accuracy scores
accuracy_dict = {}

# Set path for saving figures
save_directory = f'{base_path}/vis/concatenated/{freq}/phoneme_decoding'
os.makedirs(save_directory, exist_ok=True)

for stim in stimuli:
    for run in runs:
        # Construct file path to concatenated files
        fif_path = f'{base_path}/derivatives/concatenated/{freq}/{pro}/phoneme-epo-{stim}-{run}-epo.fif'
        phoneme_path = os.path.join(base_path, 'annotations', 'phonemes', 'tsv', f'{stim}-phonemes.tsv')

        # Check if file exists
        if not os.path.exists(fif_path):
            print(f"File does not exist: {fif_path}, skipping...")
            continue

        print(f"Processing file: {fif_path}")

        # Load the epochs
        epochs = mne.read_epochs(fif_path, preload=True)

        # Downsample the epochs
        epochs.resample(100)

        # Sklearn wrapper for Logistic Regression
        clf = make_pipeline(StandardScaler(), LogisticRegression(solver='liblinear'))
        cv = KFold(5, shuffle=True)

        # Loop through each phonetic feature
        for feat in features:
            if feat not in epochs.metadata.columns:
                print(f"Feature '{feat}' not found in metadata. Skipping...")
                continue

            fig, ax = plt.subplots(1, figsize=(10, 5))
            plt.title(f"Decoding Accuracy for {feat} ({stim} - {run})")

            # Get unique categories within the feature
            values = epochs.metadata[feat].unique()
            palette = sns.color_palette("tab10", n_colors=len(values))

            for i, value in enumerate(values):
                y = (epochs.metadata[feat] == value).astype(int)

                # Preallocate accuracy scores array
                accuracy_scores = np.empty(epochs.get_data(copy=True).shape[-1])

                # Extract data and compute ROC-AUC scores across time
                for tt in range(accuracy_scores.shape[0]):
                    X_ = epochs.get_data(copy=True)[:, :, tt]
                    scores = cross_val_score(clf, X_, y, scoring='roc_auc', cv=cv, n_jobs=-1)
                    accuracy_scores[tt] = scores.mean()

                # Plotting
                ax.plot(epochs.times, accuracy_scores, label=f'{feat}-{value}', color=palette[i])

            # Save figure
            save_path = f'{save_directory}/{stim}_{run}_{feat}_logistic.jpg'
            ax.set_xlabel("Time (ms) relative to phoneme onset")
            ax.set_ylabel("ROC-AUC")
            ax.legend()
            plt.savefig(save_path)
            plt.close(fig)

            # Save accuracy scores to a CSV file
            csv_file_path = os.path.join(base_path, f'{stim}_{run}.csv')
            with open(csv_file_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Key', 'Accuracy Scores'])
                for key, scores in accuracy_dict.items():
                    writer.writerow([key, scores])

print("Analysis completed.")