import mne
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, KFold
import os
import csv

# List all sessions, stimuli, and runs to loop through
session_stimuli = {
    'ses-01': ['Jobs1', 'Jobs2', 'Jobs3'],
    'ses-02': ['AttFast', 'AttSlow', 'BecFast', 'BecSlow', 'CampFast', 'CampSlow']
}
runs = ['run-01', 'run-02']

# List subjects to loop through
subjects = ['sub-07','sub-08']

# Select frequency
freq = '1-40Hz'

# Set base path depending on who is running the code
user_paths = ['/Users/derekrosenzweig/Documents/GitHub/iEEG-Speech',
              '/Users/lauragwilliams/Documents/projects/iEEG-Speech']

for user_path in user_paths:
    if os.path.exists(user_path):
        base_path = user_path

fig_path = os.path.join(base_path, 'vis', 'individual', freq, 'word')
os.makedirs(fig_path, exist_ok=True)

# Define the classes you want to use for decoding
classes = ['NOUN', 'VERB', 'DET']

# Loop through subjects, sessions, stimuli, and runs
for sub in subjects:
    for ses, stims in session_stimuli.items():
        for stim in stims:
            for run in runs:

                word_epo_path = f'{base_path}/derivatives/individual/{freq}/word_epochs/word-epo-{sub}-{ses}-{stim}-{run}-epo.fif'

                # Check if file exists
                if not os.path.exists(word_epo_path):
                    print(f"File does not exist: {word_epo_path}, skipping...")
                    continue

                # Load the epochs
                epochs = mne.read_epochs(word_epo_path, preload=True)

                # Downsample the epochs
                epochs.resample(25)

                # Check unique values in the target variable (Spacy_POS)
                unique_POS_tags = epochs.metadata['Spacy_POS'].unique()
                print("Unique POS tags:", unique_POS_tags)

                # Check if there are at least 2 classes for training the model
                if len(set(classes) & set(unique_POS_tags)) < 2:
                    print("Error: The data contains less than 2 desired classes, which is insufficient for model training.")
                    continue

                # Sklearn wrapper for Logistic Regression
                clf = make_pipeline(StandardScaler(), LogisticRegression(solver='liblinear'))
                cv = KFold(5, shuffle=True)

                # Define shades of blue
                blue_palette = sns.color_palette('Blues', len(classes))

                # Filter epochs based on POS features for decoding
                filtered_epochs = [epochs[epochs.metadata['Spacy_POS'] == cls] for cls in classes]

                # Create data matrix X and label vector y
                X = np.concatenate([epoch.get_data(copy=True) for epoch in filtered_epochs], axis=0)
                y = np.concatenate([np.ones(len(epoch)) * idx for idx, epoch in enumerate(filtered_epochs)])

                # Plotting
                fig, ax = plt.subplots(1, figsize=(10, 5))
                plt.title(f"Decoding Accuracy for {sub} ({freq})")

                # Preallocate accuracy scores array
                accuracy_scores = np.zeros((len(classes), X.shape[-1]))

                # Extract data and compute ROC-AUC scores across time
                for tt in range(X.shape[-1]):
                    X_t = X[:, :, tt]
                    for idx, label in enumerate(classes):
                        y_binary = (y == idx).astype(int)
                        scores = cross_val_score(clf, X_t, y_binary, scoring='roc_auc', cv=cv, n_jobs=-1)
                        accuracy_scores[idx, tt] = scores.mean()

                # Plot accuracy scores with different shades of blue
                for idx, label in enumerate(classes):
                    ax.plot(epochs.times, accuracy_scores[idx], label=label, color=blue_palette[idx])

                # Add vertical dashed grey line at t=0
                ax.axvline(x=0, color='grey', linestyle='--')

                # Add horizontal dashed line at y=0.5 (chance level decoding)
                ax.axhline(y=0.5, color='grey', linestyle='--')

                ax.set_xlabel("Time (ms) relative to phoneme onset")
                ax.set_ylabel("ROC-AUC")
                ax.legend()
                plt.savefig(f'{fig_path}/{sub}_{ses}_{stim}_{run}_logistic.jpg', dpi=300, bbox_inches='tight')
                plt.show()

