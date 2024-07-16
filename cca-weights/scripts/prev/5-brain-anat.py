import os
import numpy as np
import pandas as pd
from surfer import Brain
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, to_hex

# Set up paths
freq = '70-150Hz'
num_elecs = 1000  # Total number of electrodes

# Path to fsaverage
subjects_dir = '/freesurfer_subjects'
subject_id = 'fsaverage'

# Paths to the TXT file directories
base_dir = '/cca-weights/txt'

# Files and titles with colormaps for each analysis type
files_and_titles = [
    ('ENTROPY', 'GPT-2 Entropy', 'Blues'),  # for filepath input 'ENTROPY'
    ('F0', 'Fundamental Frequency (Hz)', 'Greens'),  # 'F0'
    ('INTENSITY', 'Intensity (dB)', 'Greens'),  # INTENSITY
    ('EMBEDDING_1', 'GPT-2 PC1', 'Blues')  # EMBEDDING for PC1
]

# List of anatomical sites to process
anatomical_sites = [
    'Superior_Temporal_Gyrus_STG',
    'Cerebral_WM',
    'Heschls_Gyrus_HG',
    'Middle_Temporal_Gyrus_MTG'
'Insula',
'Thalamus',
# 'Inferior_Frontal_Gyrus_IFG'
]

# Function to normalize weights and generate colors using a colormap
def generate_colors(weights, cmap_name):
    norm = Normalize(vmin=weights.min(), vmax=weights.max())
    cmap = plt.get_cmap(cmap_name)
    return cmap(norm(weights))

# Process electrodes for each anatomical site and hemisphere
for site in anatomical_sites:
    for analysis_type, title, cmap_name in files_and_titles:
        for hemi in ['lh', 'rh']:
            # Electrode file path
            electrode_file = os.path.join(base_dir, site, f'{site}_{analysis_type}_{freq}_{hemi}.txt')

            if not os.path.exists(electrode_file):
                print(f"File not found: {electrode_file}")
                continue

            # Load electrode positions and weights
            electrode_data = pd.read_csv(electrode_file, delim_whitespace=True, header=None, names=['x', 'y', 'z', 'LvsR', 'weight'])
            electrode_positions = electrode_data.iloc[:num_elecs, :3].values.astype(float)
            electrode_weights = np.abs(electrode_data['weight'].values[:num_elecs])  # Take absolute value of weights

            # Reflect coordinates for the right hemisphere
            if hemi == 'rh':
                electrode_positions[:, 0] = -electrode_positions[:, 0]

            # Generate colors based on weights using the colormap
            colors = generate_colors(electrode_weights, cmap_name)

            # Set up the visualization for the hemisphere
            brain = Brain(subject_id, hemi, 'inflated', subjects_dir=subjects_dir, cortex='low_contrast', alpha=0.3, background='white')

            # Add the electrode positions to the brain with colors based on weights
            for pos, col in zip(electrode_positions, colors):
                x, y, z = pos
                brain.add_foci([[x, y, z]], coords_as_verts=False, hemi=hemi, color=to_hex(col), scale_factor=0.5, alpha=1.0, name=analysis_type)

            # Save the visualization
            save_path = os.path.join(base_dir, '../..', 'new_figures', 'anat-pysurfer', f'{site}_{analysis_type}_{num_elecs}_{hemi}.png')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            brain.save_image(save_path)

            # Show the visualization
            brain.show_view('lateral')

            # Close the visualization to avoid rendering issues
            brain.close()

print("Finished plotting electrodes.")
