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


# Function to create the posterior-anterior gradient plot
def create_pa_gradient_plot(df, feature, title, output_path):
    # Filter for STG electrodes
    stg_df = df[df['Anat_Label'].isin(anatomical_sites)]

    if stg_df.empty:
        print(f"No STG electrodes found for {feature}")
        return

    print(f"Found {len(stg_df)} STG electrodes for {feature}")

    # Extract the anterior-posterior coordinate (assumed to be the y-coordinate here, adjust if different)
    stg_df['pa_coord'] = stg_df['fsaverageINF_coord_2']

    # Sort by the posterior-anterior coordinate
    stg_df = stg_df.sort_values(by='pa_coord')

    # Normalize the coordinate for color mapping
    norm = plt.Normalize(stg_df['pa_coord'].min(), stg_df['pa_coord'].max())

    # Create the color palette based on the posterior-anterior gradient
    cmap = plt.get_cmap('coolwarm')
    colors = cmap(norm(stg_df['pa_coord']))

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(stg_df['pa_coord'], stg_df['weight'], c=colors, edgecolor='k', s=100)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Posterior-Anterior Coordinate', rotation=270, labelpad=15)

    # Plot formatting
    ax.set_title(f'{title} - STG Electrode Weights on Posterior-Anterior Gradient', fontsize=16)
    ax.set_xlabel('Posterior-Anterior Coordinate', fontsize=12)
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
            output_path = os.path.join(fig_dir, f'{freq}_CCA_weights_{input_name}_STG_pa_gradient.png')
            create_pa_gradient_plot(df, input_name, title, output_path)
        else:
            print(f"File not found: {file_path}")
