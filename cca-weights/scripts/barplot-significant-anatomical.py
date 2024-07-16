import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

# Path configuration
base_path = '/Users/derekrosenzweig/PycharmProjects/CCA-reduce'
csv_dir = os.path.join(base_path, 'cca-weights', 'weights')
fig_dir = os.path.join(base_path, 'figures', 'barplots')
os.makedirs(fig_dir, exist_ok=True)

# List of files
files_and_titles = [
    ('F0', 'Fundamental Frequency', 'darkgreen'),
    ('INTENSITY', 'Sound Intensity', 'lightgreen'),
    ('EMBEDDING_1', 'GPT-2 PC1', 'navy'),
    ('EMBEDDING_2', 'GPT-2 PC2', 'darkblue'),
    ('EMBEDDING_3', 'GPT-2 PC3', 'blue'),
    ('EMBEDDING_4', 'GPT-2 PC4', 'deepskyblue'),
    ('EMBEDDING_5', 'GPT-2 PC5', 'lightblue'),  # Swapped colors
    ('ENTROPY', 'GPT-2 Entropy', 'dodgerblue')  # Swapped colors
]

# Z-score threshold for significance
z_threshold = 2

# Function to create barplot
def create_barplot(data, analysis_type, title, color, output_path, max_y):
    # Get the percentage of significant electrodes for each anatomical site
    anatomical_counts = data['Anatomical-Final'].value_counts()
    top_10 = anatomical_counts.nlargest(10)
    total_significant = data.shape[0]
    percentages = (top_10 / total_significant) * 100

    plt.figure(figsize=(12, 8))
    ax = percentages.plot(kind='bar', color=color)

    # Customize plot appearance
    plt.title(f'{title} - Distribution of Anatomical Sites', fontsize=22, fontweight='bold')
    plt.ylabel('Percentage of Significant Electrodes', fontsize=16, fontweight='bold')
    plt.ylim(0, max_y * 1.1)  # Add buffer room on the y-axis
    plt.xticks(rotation=0, ha='center', fontsize=14, fontweight='bold')
    plt.yticks(fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Remove spines for a cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Remove the x-axis label
    ax.set_xlabel('')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# Determine the maximum y-axis limit across all features
max_y = 0
for analysis_type, title, color in files_and_titles:
    file_path = os.path.join(csv_dir, f'70-150Hz_CCA_weights_{analysis_type}.csv')

    try:
        # Load the data
        data = pd.read_csv(file_path)

        # Filter data to include only significant electrodes
        z_scores = stats.zscore(data['weight'])
        data['z_score'] = z_scores
        significant_data = data[np.abs(data['z_score']) > z_threshold]

        # Get the percentage of significant electrodes for each anatomical site
        anatomical_counts = significant_data['Anatomical-Final'].value_counts()
        top_10 = anatomical_counts.nlargest(10)
        total_significant = significant_data.shape[0]
        percentages = (top_10 / total_significant) * 100

        # Update max_y if the current plot's max percentage is higher
        max_y = max(max_y, percentages.max())

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

# Iterate through the list of features and create barplots
for analysis_type, title, color in files_and_titles:
    file_path = os.path.join(csv_dir, f'70-150Hz_CCA_weights_{analysis_type}.csv')

    try:
        # Load the data
        data = pd.read_csv(file_path)

        # Filter data to include only significant electrodes
        z_scores = stats.zscore(data['weight'])
        data['z_score'] = z_scores
        significant_data = data[np.abs(data['z_score']) > z_threshold]

        # Create and save the barplot
        output_path = os.path.join(fig_dir, f'{analysis_type}_top10_barplot.png')
        create_barplot(significant_data, analysis_type, title, color, output_path, max_y)

        print(f"Barplot saved: {output_path}")

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

print("Finished creating barplots for all features.")
