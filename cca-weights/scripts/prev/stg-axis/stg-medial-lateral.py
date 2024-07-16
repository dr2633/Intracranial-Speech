import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

base_path = '/'

# Directory to save figures
fig_dir = f'{base_path}/cca-weights/new_figures/stg-gradient'
os.makedirs(fig_dir, exist_ok=True)

# Directory containing the CSV files
csv_dir = f'{base_path}/cca-weights/weights'

# List of frequency bands
freqs = ['70-150Hz']

# List of files and titles
files_and_titles = [
    ('F0', 'Fundamental Frequency'),
    ('INTENSITY', 'Sound Intensity')
]

# List of anatomical sites to process
anatomical_sites = ['Superior Temporal Gyrus (STG)']


# Function to create the medial-lateral gradient plot
def create_ml_gradient_plot(df, feature, title, output_path):
    # Filter for STG electrodes
    stg_df = df[df['Anat_Label'].isin(anatomical_sites)]

    if stg_df.empty:
        print(f"No STG electrodes found for {feature}")
        return

    print(f"Found {len(stg_df)} STG electrodes for {feature}")

    # Extract the medial-lateral coordinate (assumed to be the x-coordinate here)
    stg_df.loc[:, 'ml_coord'] = stg_df['fsaverageINF_coord_1']

    # Sort by the medial-lateral coordinate
    stg_df = stg_df.sort_values(by='ml_coord')

    # Normalize the coordinate for color mapping
    norm = plt.Normalize(stg_df['ml_coord'].min(), stg_df['ml_coord'].max())

    # Create the color palette based on the medial-lateral gradient
    cmap = plt.get_cmap('viridis')
    colors = cmap(norm(stg_df['ml_coord']))

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(stg_df['ml_coord'], stg_df['weight'], c=colors, edgecolor='k', s=100)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Medial-Lateral Coordinate', rotation=270, labelpad=15)

    # Plot formatting
    ax.set_title(f'{title} - STG Electrode Weights on Medial-Lateral Gradient', fontsize=16)
    ax.set_xlabel('Medial-Lateral Coordinate', fontsize=12)
    ax.set_ylabel('CCA Weights', fontsize=12)

    # Save the plot
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


# Loop through frequency bands
for freq in freqs:
    # Loop through files and create plots
    for input_name, title in files_and_titles:
        file_path = f'{csv_dir}/{freq}_CCA_weights_{input_name}.csv'
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            output_path = os.path.join(fig_dir, f'{freq}_CCA_weights_{input_name}_STG_ml_gradient.png')
            create_ml_gradient_plot(df, input_name, title, output_path)
        else:
            print(f"File not found: {file_path}")

#
# The medial-lateral axis is usually represented by the x-coordinate in the brain coordinate system. In the fsaverageINF coordinate system:
#
# The medial side is closer to the midline of the brain (x-coordinate near 0).
# The lateral side is further from the midline (x-coordinate is positive for the left hemisphere and negative for the right hemisphere).
# Steps to Compute Medial-Lateral Gradient
# Filter the electrodes in STG.
# Extract the medial-lateral coordinate (x-coordinate).
# Normalize the coordinates to create a color gradient.
# Plot the coordinates and weights.
