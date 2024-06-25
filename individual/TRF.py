import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
import mne
from mne.decoding import ReceptiveField, TimeDelayingRidge


# List all sessions, stimuli, and runs to loop through
session_stimuli = {
    'ses-01': ['Jobs1', 'Jobs2', 'Jobs3'],
    'ses-02': ['AttFast', 'AttSlow', 'BecFast', 'BecSlow', 'CampFast', 'CampSlow']
}
runs = ['run-01', 'run-02']

# List subjects to loop through
subjects = ['sub-07','sub-08']

# Select frequency
freq = '70-150Hz'


# Set base path depending on who is running the code
user_paths = ['/Users/derekrosenzweig/Documents/GitHub/iEEG-Speech',
              '/Users/lauragwilliams/Documents/projects/iEEG-Speech']

for user_path in user_paths:
    if os.path.exists(user_path):
        base_path = user_path


BIDS_path = os.path.join(base_path, 'BIDS')
fig_path = os.path.join(base_path, 'vis', 'individual', freq, 'TRF')
os.makedirs(fig_path, exist_ok=True)

# Loop through subjects, sessions, stimuli, and runs
for sub in subjects:
    for ses, stims in session_stimuli.items():
        for stim in stims:
            for run in runs:
                # Construct file path
                fif_path = os.path.join(base_path, 'derivatives', 'individual', freq, 'preprocessed',
                                        f'{sub}_{ses}_task-listen{stim}_{run}_ieeg.fif')

                # Check if file exists
                if not os.path.exists(fif_path):
                    print(f"File does not exist: {fif_path}, skipping...")
                    continue

                print(f"Processing file: {fif_path}")

                # Construct file paths
                word_path = os.path.join(base_path, 'annotations', 'words', 'tsv', f'{stim}-words.tsv')
                phoneme_path = os.path.join(base_path, 'annotations', 'phonemes', 'tsv', f'{stim}-phonemes.tsv')

                # define features to model
                word_features = ['word_onset', 'Duration', 'NMorph',
                                 'Phon_N', 'Word_Position', 'Sentence_Start']

                phoneme_features = ['phoneme_onset', 'phonation_v', 'manner_n', 'manner_f',
                                    'Position']


                # norming function to account for difference in sparsity across features
                def norm_(x):
                    '''
                    Norm a single feature
                    '''
                    x = (x - np.min(x)) / (np.max(x) - np.min(x))
                    return x / np.max(x)


                def norm(X):
                    '''
                    Norm a matrix of features
                    '''
                    n_features, n_times = X.shape
                    normed = np.array([norm_(X[fi, :]) for fi in range(n_features)])
                    return normed


                # scorer. here we use pearson because we are correlating two time series.
                # for the epoch-based decoding, we instead use spearman because its a rank-based correlation
                def scorer_pearson(y_true, y_pred):
                    '''
                    Compute the pearson correlation between the true timecourse and predicted
                    timecourse for each electrode in the matrix.
                    '''
                    n_times, n_features = y_pred.shape
                    r = [pearsonr(y_true[:, fi], y_pred[:, fi])[0] for fi in range(n_features)]
                    return np.array(r)


                # Load metadata for words from annotations
                word_info = pd.read_csv(word_path, delimiter='\t', encoding='latin-1')

                # Load metadata for phonemes from annotations
                phoneme_info = pd.read_csv(phoneme_path, delimiter='\t', encoding='latin-1')

                # add some additional features
                phoneme_info['phoneme_onset'] = np.ones(len(phoneme_info))
                word_info['word_onset'] = np.ones(len(word_info))
                word_info['Sentence_Start'] = (word_info['Word_Position'] == 0) * 1

                # turn the string variables into binary ones for phonemes
                manner_vals = np.unique(phoneme_info['manner'].values)
                for mannner_val in manner_vals:
                    phoneme_info['manner_%s' % (mannner_val)] = (phoneme_info['manner'].values == mannner_val) * 1
                phoneme_info['phonation_v'] = (phoneme_info['phonation'] == 'v') * 1
                phoneme_info['centrality_c'] = (phoneme_info['centrality'] == 'c') * 1

                # Read the preprocessed raw data from the .fif file
                raw_CAR = mne.io.read_raw_fif(fif_path)
                sfreq = raw_CAR.info['sfreq']

                # Get data from preprocessed -- useful to check
                CAR_z = raw_CAR.get_data()
                n_chs, n_times = CAR_z.shape

                # Turn the dataframe into a continuous timecourse of lang_feature x time,
                # where the time dimension matches the raw iEEG

                # -#-# WORDS #-#-#

                n_w_features = len(word_features)
                w_feature_timecourse = np.zeros([n_w_features, n_times])

                # loop through each row of the dataframe, and insert an impulse proportional
                # to the value of the feature at that instant
                for wi, sec_start in enumerate(word_info['Start'].values):
                    # get sample of start time
                    sample_start = int(sec_start * sfreq)

                    # extract value at this word
                    value = np.array(word_info[word_features].values[wi, :], dtype='float')

                    # if a value is a nan, just use a zero (i.e., skip)
                    finite_idx = ~np.isfinite(value)
                    value[finite_idx] = 0

                    # put into matrix
                    if sample_start >= n_times:
                        print(f"Sample start {sample_start} exceeds array bounds, skipping file...")
                        continue
                    w_feature_timecourse[:n_w_features, sample_start] = value

                # -#-# PHONEMES #-#-#

                # phoneme features
                n_p_features = len(phoneme_features)
                p_feature_timecourse = np.zeros([n_p_features, n_times])

                # loop through each row of the dataframe, and insert an impulse proportional
                # to the value of the feature at that instant
                for pi, sec_start in enumerate(phoneme_info['Start'].values):
                    # get sample of start time
                    sample_start = int(sec_start * sfreq)

                    # extract value at this word
                    value = np.array(phoneme_info[phoneme_features].values[pi, :], dtype='float')

                    # if a value is a nan, just use a zero (i.e., skip)
                    finite_idx = ~np.isfinite(value)
                    value[finite_idx] = 0

                    # put into matrix
                    if sample_start >= n_times:
                        print(f"Sample start {sample_start} exceeds array bounds, skipping file...")
                        continue
                    p_feature_timecourse[:n_p_features, sample_start] = value

                # concatenate the word and phoneme features
                feature_timecourse = np.concatenate([p_feature_timecourse, w_feature_timecourse])
                features = word_features + phoneme_features
                n_features = len(features)

                # it is important to norm all the features. otherwise we cannot directly
                # compare coefficient magnitude
                normed_features = norm(feature_timecourse)

                # init X,y. in an encoding model, Y is the brain data and X are the stim features
                y = CAR_z
                X = normed_features

                # define train/test indices based on a proportion. so we will train on the
                # first X% of the data and test on the remaining X% of the data
                proportion_train = 0.7
                recoding_duration = raw_CAR.times[-1]
                train_idx = raw_CAR.times < recoding_duration * proportion_train
                test_idx = raw_CAR.times >= recoding_duration * proportion_train

                # for readibility, subset the X and y into their train and test
                # and transpose them into shape times x features
                y_train = y[:, train_idx].T
                X_train = X[:, train_idx].T
                y_test = y[:, test_idx].T
                X_test = X[:, test_idx].T

                # define the model
                tmin, tmax = -0.5, 0.5
                alpha = 50

                estimator = TimeDelayingRidge(tmin, tmax, sfreq, reg_type="laplacian", alpha=alpha)
                rf = ReceptiveField(tmin, tmax, sfreq, features, estimator=estimator, n_jobs=-1)

                # fit the model
                rf.fit(X_train, y_train)

                # Now make predictions about the model output, given input stimuli
                # and compute the encoding performance, where here we use a correlation
                # between the true electrode timecourse and the predicted electrode timecourse
                y_pred = rf.predict(X_test)
                scores = scorer_pearson(y_pred, y_test)

                # get the coefs for all elecs
                coefs = rf.coef_

                # an ordered list of the best electrodes
                best_elecs = np.argsort(scores)[::-1]

                # Create a DataFrame containing electrode names and their corresponding scores
                elec_df = pd.DataFrame({
                    'Electrode': [raw_CAR.info['ch_names'][elec] for elec in best_elecs],
                    'Score': scores[best_elecs]
                })

                # Select electrodes with the highest accuracy score (e.g., top 10)
                top_elec_df = elec_df

                # Define the parent directory for saving figures and TSV files
                parent_fig_path = os.path.join(base_path, 'vis', 'individual', freq, 'TRF', sub)

                # Create directories for saving figures if they don't exist
                for directory in ['hist', 'predict', 'coefs', 'tsv']:
                    fig_dir = os.path.join(parent_fig_path, directory)
                    os.makedirs(fig_dir, exist_ok=True)

                elec_tsv_path = os.path.join(parent_fig_path, 'tsv',
                                             f'{sub}_{stim}_{run}_{freq}_top_electrodes.tsv')

                # Save the DataFrame to a TSV file
                top_elec_df.to_csv(elec_tsv_path, sep='\t', index=False)

                # plot histogram of scores
                plt.hist(scores, bins=50, alpha=0.8, facecolor='teal')
                plt.xlabel("Pearson R Encoding Score")
                plt.ylabel("Number of Electrodes")
                plt.title("%s %s %s" % (sub, stim, freq))
                plt.savefig(os.path.join(parent_fig_path, 'hist',
                                         f'{sub}_{stim}_{run}_{freq}_histogram.png'))
                plt.show()

                # plot true and predictions for some good elecs
                fig, axs = plt.subplots(4, 2, figsize=(20, 8))
                axs = axs.flatten()
                for ei, elec in enumerate(best_elecs[::3][:len(axs)]):
                    axs[ei].plot(y_pred[20000:40000, elec], label='y_pred', alpha=1, c='k')
                    axs[ei].plot(y_test[20000:40000, elec], label='y_true', alpha=0.4)
                    title = '%s r=%s' % (raw_CAR.info['ch_names'][elec], np.round(scores[elec], 2))
                    axs[ei].set_title(title)
                    axs[ei].axis('off')
                plt.legend()
                plt.savefig(os.path.join(parent_fig_path, 'predict',
                                         f'{sub}_{stim}_{run}_{freq}_predictions.png'))
                plt.show()

                # plot coefs for some good elecs
                fig, axs = plt.subplots(4, 2, figsize=(8, 9))
                axs = axs.flatten()
                vlim = coefs.max() * 0.7
                for ei, elec in enumerate(best_elecs[::3][:len(axs)]):
                    axs[ei].matshow(coefs[elec, ...], aspect=100, cmap='RdBu_r',
                                    vmin=-vlim, vmax=vlim)
                    axs[ei].vlines(x=500, ymin=-0.5, ymax=n_features - 0.5, ls='--', color='k')
                    axs[ei].set_yticks(np.arange(n_features), labels=features)
                plt.savefig(os.path.join(parent_fig_path, 'coefs', f'{sub}_{freq}_coefs.png'))
                plt.show()
