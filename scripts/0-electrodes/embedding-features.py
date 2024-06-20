# import modules
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from wordfreq import word_frequency
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, make_scorer, get_scorer, roc_auc_score
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score, KFold
import glob
import os
import seaborn as sns
import ast

# Set base path
base_path = '/Users/derekrosenzweig/Documents/GitHub/iEEG-Speech'

# Define the directory path
save_directory = '/Users/derekrosenzweig/Documents/GitHub/ieeg/iEEG_BIDS/sEEG_BIDS/vis/word/context/'
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# params
sub = 'sub-06'
ses = 'ses-01'
stim = 'Jobs1'
run = 'run-01'
freq = "70-150Hz"

word_path = f'{base_path}/derivatives/{freq}/word_epochs/word-epo-{sub}-{ses}-{stim}-{run}-epo.fif'

epochs = mne.read_epochs(word_path, preload=True)

# Get the content and shape of the epoched data
print(epochs.info)
print(epochs.metadata)
print(np.shape(epochs))

# Extract epoched iEEG data
ieeg_data = epochs.get_data()

# Extract GPT-2 embeddings for multiple layers
layers = ['Layer_8']

for layer in layers:
    print(f"Processing {layer}...")

    # Extract GPT-2 embeddings for the current layer
    gpt_embeddings_str = epochs.metadata[layer].values

    # Convert string embeddings to float arrays
    gpt_embeddings = np.array([np.array(ast.literal_eval(emb)) for emb in gpt_embeddings_str])

    print(f"GPT-2 embeddings shape for {layer}: {gpt_embeddings.shape}")

    # Reshape the iEEG data to have 2 dimensions (samples, features)
    ieeg_data_reshaped = ieeg_data.reshape(ieeg_data.shape[0], -1)

    X_train, X_test, y_train, y_test = train_test_split(gpt_embeddings, ieeg_data_reshaped, test_size=0.2,
                                                        random_state=42)

    # Check shapes of train and test sets
    print(f"Training set shapes for {layer}:", X_train.shape, y_train.shape)
    print(f"Testing set shapes for {layer}:", X_test.shape, y_test.shape)

    # Initialize the Ridge regression model
    ridge = Ridge(alpha=1.0)  # You can adjust the regularization parameter alpha as needed

    # Fit the model
    ridge.fit(X_train, y_train)

    # Evaluate the model on the test set
    score = ridge.score(X_test, y_test)
    print(f"Ridge Regression Score for {layer}:", score)