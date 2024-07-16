import os
import numpy as np
import pandas as pd
from surfer import Brain
import scipy.stats as stats

base_path = '/Users/derekrosenzweig/PycharmProjects/CCA-reduce'
freq = '70-150Hz'
num_elecs = 1000  # Total number of electrodes
z_threshold = 2  # Z-score threshold for significance

# Path to fsaverage
subjects_dir = os.path.join(base_path, 'freesurfer_subjects')
subject_id = 'fsaverage'

# Updated path to the LH electrode file directory
lh_dir = os.path.join(base_path, 'cca-weights/txt')

# List of files and titles with their corresponding colors
files_and_titles = [
    ('F0', 'F0 (Hz)', 'darkgreen'),
    ('INTENSITY', 'Intensity (dB)', 'lightgreen'),
    ('EMBEDDING_1', 'GPT-2 PC1', 'navy'),
    ('EMBEDDING_2', 'GPT-2 PC2', 'darkblue'),
    ('EMBEDDING_3', 'GPT-2 PC3', 'blue'),
    ('EMBEDDING_4', 'GPT-2 PC4', 'deepskyblue'),
    ('EMBEDDING_5', 'GPT-2 PC5', 'lightblue'),
    ('ENTROPY', 'GPT-2 Entropy', 'dodgerblue')
]

# Function to load and process electrode data
def load_electrode_data(lh_file_path, color):
    try:
        # Load LH data
        lh_data = pd.read_csv(lh_file_path, delim_whitespace=True, header=None, names=['x', 'y', 'z', 'LvsR', 'weight', 'WMvsGM'])

        # Compute z-scores across LH data
        z_scores = stats.zscore(lh_data['weight'])
        lh_data['z_score'] = z_scores

        # Filter significant electrodes based on z-scores
        significant_data = lh_data[np.abs(lh_data['z_score']) > z_threshold]
        significant_data['color'] = color  # Assign color to significant electrodes

        return lh_data, significant_data[['x', 'y', 'z', 'color']]
    except Exception as e:
        print(f"Error loading data from {lh_file_path}: {e}")
        return pd.DataFrame(), pd.DataFrame()

# Function to plot electrodes for the LH hemisphere
def plot_electrodes(electrode_data, significant_positions, save_path):
    brain = Brain(subject_id, 'lh', 'inflated', subjects_dir=subjects_dir, cortex='low_contrast', alpha=0.3, background='white')

    # Plot all electrodes with reduced transparency
    for _, row in electrode_data.iterrows():
        x, y, z = row[['x', 'y', 'z']]
        brain.add_foci([[x, y, z]], coords_as_verts=False, hemi='lh', color='gray', scale_factor=0.5, alpha=0.1, name='all_elecs')

    # Plot significant electrodes with specific colors
    for _, row in significant_positions.iterrows():
        x, y, z, color = row[['x', 'y', 'z', 'color']]
        brain.add_foci([[x, y, z]], coords_as_verts=False, hemi='lh', color=color, scale_factor=0.5, alpha=1.0, name='significant_elecs')

    # Save the visualization
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    brain.save_image(save_path)

    # Show the visualization
    brain.show_view('lateral')

    # Close the visualization to avoid rendering issues
    brain.close()

# Iterate through the list of features and colors
for analysis_type, title, color in files_and_titles:
    # Electrode file path for LH
    lh_file_path = os.path.join(lh_dir, f'top_{num_elecs}_electrodes_{analysis_type}_{freq}_lh.txt')

    # Load electrode positions and weights
    lh_data, lh_significant = load_electrode_data(lh_file_path, color)

    # Save path for visualization
    lh_save_path = os.path.join(base_path, 'figures/surfer-significant', f'{freq}_significant_electrodes_{analysis_type}_lh.png')

    # Plot electrodes for LH
    plot_electrodes(lh_data, lh_significant, lh_save_path)

print("Finished plotting significant electrodes for all features.")
