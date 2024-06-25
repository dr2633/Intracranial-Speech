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
sub = 'sub-02'
ses = 'ses-01'
freq = '1-40Hz'
loop_over = 'time'

# Define the directory path for saving figures
save_directory = f'{base_path}/vis/individual/embeddings'
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# Load file names for a subject
search_expression = f'{base_path}/derivatives/individual/{freq}/word_epochs/word-epo-{sub}-{ses}-*-*-epo.fif'
files = glob.glob(search_expression)

if not files:
    print("No epoch files found matching the search expression.")
else:
    # concatenate
    epoch_list = []
    for f in files:
        epoch = mne.read_epochs(f, preload=True)
        if epoch is not None:
            epoch_list.append(epoch)
        else:
            print(f"Failed to load epoch from {f}")

    if not epoch_list:
        print("No valid epochs loaded.")
    else:
        # Concatenate epochs
        epochs = mne.concatenate_epochs(epoch_list)

        # Downsample the epochs
        epochs.resample(50)

        # Print the number of epochs
        num_epochs = len(epochs)
        print(f"Number of epochs: {num_epochs}")

clf = make_pipeline(StandardScaler(), Ridge(alpha=10000.))  # Specify alpha value for Ridge regression
scorer = make_scorer(scorer_spearman)
cv = KFold(5, shuffle=True)

if loop_over == 'time':
    features = ['Layer_4', 'Layer_8', 'Layer_10','Layer_12']
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
            s = cross_val_score(clf, gpt_embeddings, X_reshaped.mean(axis=1), scoring=scorer, cv=cv, n_jobs=-1).mean()
            spearman_scores[tt] = s
            print(feature, tt)

        ax.plot(epochs.times, spearman_scores, label=feature, color=colors(i))

    ax.set_ylim([-0.05, 0.2])
    ax.set_xlabel("Time (ms) relative to word onset", fontsize=14)
    ax.set_ylabel("Spearman R", fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.axvline(x=0, color='grey', linestyle='--', linewidth=1, alpha=0.7)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    plt.title(f"Ridge Regression Spearman R for {sub} ({ses})", fontsize=16)
    plt.legend(fontsize=10, loc='upper right')

    # Save the figure for the current feature
    save_path = f'{save_directory}/{sub}_{ses}_{freq}_embeddings.jpg'
    plt.savefig(save_path, dpi=300)
    plt.show()