import os
import numpy as np
import pandas as pd
from surfer import Brain

base_path = '/'

# Directory to save figures
fig_dir = f'{base_path}/figures/custom-colors'
os.makedirs(fig_dir, exist_ok=True)

# Directory containing the CSV files
csv_dir = f'{base_path}/cca-weights/weights'

# List of frequency bands
freqs = ['70-150Hz']

# List of files and titles with their corresponding colors
files_and_titles = [
    ('EMBEDDING_1', 'GPT-2 PC1', 'navy'),
    ('EMBEDDING_2', 'GPT-2 PC2', 'darkblue'),
    ('EMBEDDING_3', 'GPT-2 PC3', 'blue'),
    ('EMBEDDING_4', 'GPT-2 PC4', 'deepskyblue'),
    ('EMBEDDING_5', 'GPT-2 PC5', 'dodgerblue'),
    ('ENTROPY', 'GPT-2 Entropy', 'lightblue'),
    ('F0', 'Fundamental Frequency', 'lightgreen'),
    ('INTENSITY', 'Sound Intensity', 'darkgreen')
]

# Path to fsaverage
subjects_dir = os.path.join(base_path, 'freesurfer_subjects')
subject_id = 'fsaverage'

# Updated paths to the LH and RH electrode file directories
lh_dir = os.path.join(base_path, 'cca-weights/txt')
rh_dir = os.path.join(base_path, 'cca-weights/txt')

# Function to load and process electrode data
def load_electrode_data(file_path, top_n):
    try:
        electrode_data = pd.read_csv(file_path, delim_whitespace=True, header=None, names=['x', 'y', 'z', 'name', 'weight', 'WMvsGM'])
        electrode_data_sorted = electrode_data.sort_values(by='weight', ascending=False).head(top_n)
        electrode_positions = electrode_data_sorted[['x', 'y', 'z']].values.astype(float)
        return electrode_positions
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return np.array([])

# Process electrodes for both hemispheres and all features
for freq in freqs:
    for analysis_type, title, color in files_and_titles:
        # Initialize Brain visualization for the current feature group
        brain = Brain(subject_id, 'both', 'inflated', subjects_dir=subjects_dir, cortex='low_contrast', alpha=0.3, background='white')

        for hemi in ['lh', 'rh']:
            # Electrode file path
            if hemi == 'lh':
                electrode_file = os.path.join(lh_dir, f'top_{top_n}_electrodes_{analysis_type}_{freq}_lh.txt')
            else:
                electrode_file = os.path.join(rh_dir, f'top_{top_n}_electrodes_{analysis_type}_{freq}_rh.txt')

            print(f"Processing file: {electrode_file}")

            # Load top N electrode positions
            electrode_positions = load_electrode_data(electrode_file, top_n)

            if electrode_positions.size == 0:
                continue

            print(f"Loaded positions before reflection for {analysis_type} ({hemi}):")
            print(electrode_positions)

            # Reflect coordinates for the right hemisphere
            if hemi == 'rh':
                electrode_positions[:, 0] = -electrode_positions[:, 0]

            print(f"Loaded positions after reflection for {analysis_type} ({hemi}):")
            print(electrode_positions)

            # Add the electrode positions to the brain
            for pos in electrode_positions:
                x, y, z = pos
                brain.add_foci([[x, y, z]], coords_as_verts=False, hemi=hemi, color=color, scale_factor=0.7, alpha=1.0, name=f"{analysis_type}_{hemi}")

        # Save the visualization
        save_path = os.path.join(fig_dir, f'{freq}_top_10_electrodes_{analysis_type}.png')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        brain.save_image(save_path)

        # Show the visualization
        brain.show_view('lateral')

        # Close the visualization to avoid rendering issues
        brain.close()

print("Finished plotting top 10 electrodes for each feature.")
