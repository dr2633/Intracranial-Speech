import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

base_path = '/Users/derekrosenzweig/PycharmProjects/CCA-reduce'
freq = '70-150Hz'
z_threshold = 2  # Z-score threshold for significance

# Directory to save figures and statistics
fig_dir = f'{base_path}/figures/histograms'
os.makedirs(fig_dir, exist_ok=True)
stats_dir = f'{base_path}/figures/statistics'
os.makedirs(stats_dir, exist_ok=True)
stats_csv_path = os.path.join(stats_dir, f'{freq}_statistics.csv')

# Directory containing the CSV files
csv_dir = f'{base_path}/cca-weights/weights'

# List of files and titles with their corresponding colors
files_and_titles = [
    ('F0', 'Fundamental Frequency', 'darkgreen'),
    ('INTENSITY', 'Sound Intensity', 'lightgreen'),
    ('EMBEDDING_1', 'GPT-2 PC1', 'navy'),
    ('EMBEDDING_2', 'GPT-2 PC2', 'darkblue'),
    ('EMBEDDING_3', 'GPT-2 PC3', 'blue'),
    ('EMBEDDING_4', 'GPT-2 PC4', 'deepskyblue'),
    ('EMBEDDING_5', 'GPT-2 PC5', 'lightblue'),
    ('ENTROPY', 'GPT-2 Entropy', 'dodgerblue')
]

# Initialize a list to collect statistics
all_stats = []

# Function to load and process electrode data
def load_electrode_data(file_path, z_threshold):
    try:
        electrode_data = pd.read_csv(file_path)
        z_scores = stats.zscore(electrode_data['weight'])
        electrode_data['z_score'] = z_scores
        significant_data = electrode_data[np.abs(electrode_data['z_score']) > z_threshold]
        return electrode_data, significant_data
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return pd.DataFrame(), pd.DataFrame()

# Function to create histogram with seaborn and enhanced aesthetics
def create_histogram(electrode_data, significant_data, title, color, output_path_png):
    plt.figure(figsize=(12, 8))
    sns.set(style="whitegrid")

    # Calculate bin edges for consistent bin sizes
    bins = np.linspace(-0.4, 0.4, 50)

    # Histogram for all data
    sns.histplot(electrode_data['weight'], bins=bins, color=color, alpha=0.2, kde=False, label='Non-Significant Electrodes')

    # Histogram for significant data
    sns.histplot(significant_data['weight'], bins=bins, color=color, alpha=1.0, kde=False, label='Significant Electrodes')

    plt.axvline(x=0, color='grey', linestyle='--')
    plt.title(f'Electrode Weight Distribution - {title}', fontsize=22, fontweight='bold')
    plt.xlabel('Obtained CCA Weights', fontsize=16)
    plt.ylabel('Number of Electrodes', fontsize=16)
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_path_png)
    plt.close()

# Iterate through the list of features and colors
for analysis_type, title, color in files_and_titles:
    file_path = os.path.join(csv_dir, f'{freq}_CCA_weights_{analysis_type}.csv')

    # Load electrode positions and weights
    electrode_data, significant_data = load_electrode_data(file_path, z_threshold)

    # Calculate statistics
    mean_weight = electrode_data['weight'].mean()
    std_weight = electrode_data['weight'].std()
    skew_weight = stats.skew(electrode_data['weight'])
    range_weight = electrode_data['weight'].max() - electrode_data['weight'].min()
    threshold_high = mean_weight + 2 * std_weight
    threshold_low = mean_weight - 2 * std_weight
    num_significant = len(significant_data)
    num_non_significant = len(electrode_data) - num_significant
    total_electrodes = len(electrode_data)

    # Save statistics in a dictionary
    stats_dict = {
        'Feature': title,
        'Mean': mean_weight,
        'Standard Deviation': std_weight,
        'Skewness': skew_weight,
        'Range': range_weight,
        'Threshold (+2 SD)': threshold_high,
        'Threshold (-2 SD)': threshold_low,
        'Significant Electrodes': num_significant,
        'Non-Significant Electrodes': num_non_significant,
        'Total Electrodes': total_electrodes
    }
    all_stats.append(stats_dict)

    # Save paths for histogram
    output_path_png = os.path.join(fig_dir, f'{freq}_histogram_{analysis_type}.png')

    # Create histogram
    create_histogram(electrode_data, significant_data, title, color, output_path_png)

# Save all statistics to a CSV file
stats_df = pd.DataFrame(all_stats)
stats_df.to_csv(stats_csv_path, index=False)

print("Finished creating histograms and saving statistics for all features.")
