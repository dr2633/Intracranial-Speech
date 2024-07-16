import pandas as pd
import matplotlib.pyplot as plt
import os

base_path = '/'

# Directory to save figures
fig_dir = f'{base_path}/0-PC-interpretability/figures/stem'
os.makedirs(fig_dir, exist_ok=True)

# Directory containing the CSV files
csv_dir = f'{base_path}/cca-weights/weights'

# List of frequency bands
freqs = ['70-150Hz']

# List of files and titles with colors for both frequency bands
files_and_titles = [
    ('EMBEDDING', 'GPT-2 PC1', 'navy'),
    ('EMBEDDING_2', 'GPT-2 PC2',  'darkblue'),
    ('EMBEDDING_3', 'GPT-2 PC3', 'blue'),
    ('EMBEDDING_4', 'GPT-2 PC4', 'deepskyblue'),
    ('EMBEDDING_5', 'GPT-2 PC5', 'dodgerblue'),
    ('ENTROPY', 'GPT-2 Entropy', 'lightblue'),

]

# Function to normalize the weights to a proportion (0 to 1)
def normalize_weights(weights):
    return weights / weights.sum()

# Function to compute summary statistics
def compute_summary_statistics(df):
    stats = {
        'Mean': df['normalized_weight'].mean(),
        'StdDev': df['normalized_weight'].std(),
        'Max': df['normalized_weight'].max(),
        'Min': df['normalized_weight'].min()
    }
    return stats

# Function to compute statistics by anatomical locations
def compute_stats_by_anatomy(df):
    return df.groupby('Anat')['normalized_weight'].agg(['mean', 'std', 'max', 'min'])

# Loop through frequency bands
for freq in freqs:
    # Step 1: Find the maximum y-value across all normalized weights for this frequency band
    max_y_value = 0
    for input_name, title, color in files_and_titles:
        file_path = f'{csv_dir}/{freq}_CCA_weights_{input_name}.csv'
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            normalized_weights = normalize_weights(df['weight'].abs())
            max_y_value = max(max_y_value, normalized_weights.max())
        else:
            print(f"File not found: {file_path}")

    # Add 0.001 to the maximum y-value
    max_y_value += 0.001



    # Step 2: Create the plots using the found max_y_value
    # Function to create a stem plot for normalized weights
    def create_stem_plot_normalized(ax, df, title, color):
        # Normalize the weights
        df['normalized_weight'] = normalize_weights(df['weight'].abs())

        # Compute and display summary statistics
        stats = compute_summary_statistics(df)
        print(f"Summary Statistics for {title} ({freq}):")
        print(stats)

        # Optionally, compute statistics by anatomical locations if 'Anat' column exists
        if 'Anat' in df.columns:
            anatomy_stats = compute_stats_by_anatomy(df)
            print(f"Statistics by Anatomy for {title} ({freq}):")
            print(anatomy_stats)

        # Plot the stem plot
        markerline, stemlines, baseline = ax.stem(df.index, df['normalized_weight'], linefmt=color, markerfmt='o', basefmt=" ")

        plt.setp(stemlines, 'color', color)
        plt.setp(markerline, 'color', color)

        # Title and labels
        ax.set_title(f'{title} - Normalized Weights', fontsize=18, fontweight='bold')
        ax.set_xlabel('Electrodes', fontsize=12)
        ax.set_ylabel('Normalized Absolute CCA Weights (Proportion)', fontsize=12)

        # Set the y-limit using the maximum y-value
        ax.set_ylim(0, max_y_value)

        # Add vertical dashed line at the top 100 electrodes
        ax.axvline(x=100, color='grey', linestyle='--', linewidth=1)

        # Add solid black line for x-axis
        ax.axhline(y=0, color='black', linewidth=1)

    # Create the subplot figure with 2 rows and 3 columns
    fig, axs = plt.subplots(2, 3, figsize=(20, 16))

    # Plot GPT-2 PCs and Entropy on the subplots
    for ax, (input_name, title, color) in zip(axs.flat, files_and_titles):
        file_path = f'{csv_dir}/{freq}_CCA_weights_{input_name}.csv'
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            create_stem_plot_normalized(ax, df, title, color)
        else:
            print(f"File not found: {file_path}")

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f'CCA_weights_subplot_{freq}.png'))
    plt.show()

    print(f"Figure saved to {os.path.join(fig_dir, f'CCA_weights_subplot_{freq}.png')}")

