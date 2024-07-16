import os
import numpy as np
import pandas as pd
from surfer import Brain
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm

# Set up paths
freq = '70-150Hz'
num_elecs = 1000  # Total number of electrodes

# Path to fsaverage
subjects_dir = '/Users/derekrosenzweig/Documents/GitHub/CCA-Decoding/freesurfer_subjects'
subject_id = 'fsaverage'

# Paths to the LH and RH electrode file directories
lh_dir = '/Users/derekrosenzweig/Documents/GitHub/CCA-Decoding/FS_scripts/split_txt'
rh_dir = '/Users/derekrosenzweig/Documents/GitHub/CCA-Decoding/FS_scripts/split_txt'

# Files and titles with colormaps for each analysis type
files_and_titles = [
    ('ENTROPY', 'GPT-2 Entropy'),  # for filepath input 'ENTROPY'
    ('F0', 'Fundamental Frequency (Hz)'),  # 'F0'
    ('INTENSITY', 'Intensity (dB)'),  # INTENSITY
    ('EMBEDDING', 'GPT-2 PC1')  # EMBEDDING for PC1
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
        electrode_data = pd.read_csv(electrode_file, delim_whitespace=True, header=None, names=['x', 'y', 'z', 'name', 'weight'])
        electrode_positions = electrode_data.iloc[:num_elecs, :3].values.astype(float)
        electrode_weights = electrode_data['weight'].values[:num_elecs]  # Use original weights to preserve sign

        # Reflect coordinates for the right hemisphere
        if hemi == 'rh':
            electrode_positions[:, 0] = -electrode_positions[:, 0]

        # Generate colors based on weights using a diverging colormap
        norm = TwoSlopeNorm(vmin=-np.max(np.abs(electrode_weights)), vcenter=0.0, vmax=np.max(np.abs(electrode_weights)))
        cmap = plt.get_cmap('coolwarm')
        colors = cmap(norm(electrode_weights))

        # Highlight electrodes with maximum contributions
        max_contributor_indices = np.where(np.abs(electrode_weights) == np.max(np.abs(electrode_weights)))[0]
        highlight_colors = ['gold' if i in max_contributor_indices else c for i, c in enumerate(colors)]

        # Set up the visualization for the hemisphere
        brain = Brain(subject_id, hemi, 'inflated', subjects_dir=subjects_dir, cortex='low_contrast', alpha=0.3, background='white')

        # Add the electrode positions to the brain with colors based on weights
        for pos, col in zip(electrode_positions, highlight_colors):
            x, y, z = pos
            brain.add_foci([[x, y, z]], coords_as_verts=False, hemi=hemi, color=col, scale_factor=1.0 if col == 'gold' else 0.5, alpha=1.0, name=analysis_type)

        # Save the visualization
        save_path = os.path.join('/Users/derekrosenzweig/Documents/GitHub/CCA-Decoding/figures/pysurfer2', f'{analysis_type}_{num_elecs}_{hemi}.png')
        brain.save_image(save_path)

        # Show the visualization
        brain.show_view('lateral')

        # Close the visualization to avoid rendering issues
        brain.close()

print("Finished plotting electrodes.")
