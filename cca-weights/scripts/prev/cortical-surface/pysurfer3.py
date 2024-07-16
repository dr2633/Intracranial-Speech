import os
import numpy as np
import pandas as pd
from surfer import Brain
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

base_path = '/'

# Set up paths
freq = '70-150Hz'
num_elecs = 1000  # Total number of electrodes
top_n = 100  # Number of top electrodes to highlight

# Path to fsaverage
subjects_dir = os.path.join(base_path, 'freesurfer_subjects')
subject_id = 'fsaverage'

# Paths to the LH and RH electrode file directories
lh_dir = os.path.join(base_path, 'cca-weights/txt/features')
rh_dir = os.path.join(base_path, 'cca-weights/txt/features')

# Files and titles with colormaps for each analysis type
files_and_titles = [
    ('ENTROPY', 'GPT-2 Entropy'),  # for filepath input 'ENTROPY'
    ('F0', 'Fundamental Frequency (Hz)'),  # 'F0'
    ('INTENSITY', 'Intensity (dB)'),  # INTENSITY
    ('EMBEDDING_1', 'GPT-2 PC1')  # EMBEDDING for PC1
]

# Process electrodes for both hemispheres
hemispheres = ['lh', 'rh']
for hemi in hemispheres:
    for analysis_type, title in files_and_titles:
        # Electrode file path
        if hemi == 'lh':
            electrode_file = os.path.join(lh_dir, f'top_{num_elecs}_electrodes_{analysis_type}_{freq}_lh.txt')
        else:
            electrode_file = os.path.join(rh_dir, f'top_{num_elecs}_electrodes_{analysis_type}_{freq}_rh.txt')

        # Load electrode positions and weights
        electrode_data = pd.read_csv(electrode_file, delim_whitespace=True, header=None,
                                     names=['x', 'y', 'z', 'name', 'weight', 'WMvsGM'])
        electrode_positions = electrode_data.iloc[:num_elecs, :3].values.astype(float)
        electrode_weights = electrode_data['weight'].values[:num_elecs]

        # Reflect coordinates for the right hemisphere
        if hemi == 'rh':
            electrode_positions[:, 0] = -electrode_positions[:, 0]

        # Normalize weights
        norm = Normalize(vmin=-max(abs(electrode_weights)), vmax=max(abs(electrode_weights)))

        # Create color palettes based on the weights and WMvsGM
        palette_pos = sns.color_palette("Blues", as_cmap=True)
        palette_neg = sns.color_palette("Reds", as_cmap=True)

        colors = []
        positions = []
        alphas = []
        scale_factors = []

        for i, row in electrode_data.iterrows():
            weight = row['weight']
            if weight > 0:
                color = palette_pos(norm(weight))[:3]  # Remove alpha value
            else:
                color = palette_neg(norm(weight))[:3]  # Remove alpha value

            colors.append(color)
            positions.append([row['x'], row['y'], row['z']])
            # Increase transparency for electrodes outside the top 100
            alpha = 1.0 if i < top_n else 0.3
            alphas.append(alpha)
            # Use different scale factors for WM and GM
            scale_factor = 0.8 if row['WMvsGM'] == 'WM' else 0.5
            scale_factors.append(scale_factor)

        # Set up the visualization for the hemisphere
        brain = Brain(subject_id, hemi, 'inflated', subjects_dir=subjects_dir, cortex='low_contrast', alpha=0.3,
                      background='white')

        # Add the electrode positions to the brain with colors based on weights
        for pos, col, alpha, scale in zip(positions, colors, alphas, scale_factors):
            x, y, z = pos
            brain.add_foci([[x, y, z]], coords_as_verts=False, hemi=hemi, color=col, scale_factor=scale, alpha=alpha,
                           name=analysis_type)

        # Save the visualization
        save_path = os.path.join(base_path, 'figures/pysurfer3', f'{analysis_type}_{num_elecs}_{hemi}.png')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        brain.save_image(save_path)

        # Show the visualization
        brain.show_view('lateral')

        # Close the visualization to avoid rendering issues
        brain.close()

print("Finished plotting electrodes.")
