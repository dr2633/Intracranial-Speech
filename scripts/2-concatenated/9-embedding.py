import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import os
from wordfreq import word_frequency
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, make_scorer, get_scorer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score, KFold
import glob
import ast
import matplotlib.cm as cm

base_path = '/Users/derekrosenzweig/Documents/GitHub/iEEG-Speech'

# Make SpearmanR into a decoding metric that Sklearn can read
def scorer_spearman(y_true, y_pred):
    r, _ = spearmanr(y_true, y_pred)
    return r

# params
freq = '70-150Hz'
loop_over = 'time'

# Define the directory path for saving figures
save_directory = f'{base_path}/vis/concatenated/{freq}/embeddings'
if not os.path.exists(save_directory):
    os.makedirs(save_directory)


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


for stim in stimuli:
    for run in runs:

        # Construct file path to concatenated files
        fif_path = f'{base_path}/derivatives/concatenated/{freq}/word_epochs/word-epo-{stim}-{run}-epo.fif'
        word_path = os.path.join(base_path, 'annotations', 'words', 'tsv', f'{stim}-words.tsv')

        # Check if file exists
        if not os.path.exists(fif_path):
            print(f"File does not exist: {fif_path}, skipping...")
            continue

        print(f"Processing file: {fif_path}")

        # Load the epochs
        epochs = mne.read_epochs(fif_path, preload=True)

        # Downsample the epochs
        epochs.resample(25)

        clf = make_pipeline(StandardScaler(), Ridge(alpha=100000.))
        scorer = make_scorer(scorer_spearman)
        cv = KFold(5, shuffle=True)

        if loop_over == 'time':
            features = ['Layer_8', 'Layer_10']
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))

            colors = cm.get_cmap('tab10', len(features))

            for i, feature in enumerate(features):
                gpt_embeddings_str = epochs.metadata[feature].values
                gpt_embeddings = np.array([np.array(ast.literal_eval(emb)) for emb in gpt_embeddings_str])
                print(f"GPT-2 embeddings shape for {feature}: {gpt_embeddings.shape}")

                X = epochs._data

                n_timepoints = X.shape[-1]
                spearman_scores = np.zeros(n_timepoints)
                for tt in range(n_timepoints):
                    X_ = X[:, :, tt]
                    X_reshaped = X_.reshape(X_.shape[0], -1)  # Reshape iEEG data to (n_samples, n_features)
                    s = cross_val_score(clf, gpt_embeddings, X_reshaped.mean(axis=1), scoring=scorer, cv=cv,
                                        n_jobs=-1).mean()
                    spearman_scores[tt] = s
                    print(feature, tt)

                ax.plot(epochs.times, spearman_scores, label=feature, color=colors(i))

            ax.set_ylim([-0.05, 0.2])
            ax.set_xlabel("Time (ms) relative to word onset", fontsize=14)
            ax.set_ylabel("Spearman R", fontsize=14)
            ax.tick_params(axis='both', labelsize=12)
            ax.axvline(x=0, color='grey', linestyle='--', linewidth=1, alpha=0.7)
            ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

            plt.title(f"Ridge Regression Spearman R for {stim} {run})", fontsize=16)
            plt.legend(fontsize=10, loc='upper right')

            # Save the figure for the current feature
            save_path = f'{save_directory}/{stim}_{run}_{freq}_embeddings.jpg'
            plt.savefig(save_path, dpi=300)
            plt.show()
