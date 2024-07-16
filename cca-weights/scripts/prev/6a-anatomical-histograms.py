import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix

# Set paths
base_path = '/'
fig_dir = f'{base_path}/figures/anatomical_analysis'
os.makedirs(fig_dir, exist_ok=True)
csv_dir = f'{base_path}/cca-weights/weights'
freqs = ['70-150Hz']

# Define files and titles
files_and_titles = [
    ('F0', 'F0 (Hz)', 'darkgreen'),
    ('INTENSITY', 'Intensity (dB)', 'lightgreen'),
    ('EMBEDDING', 'GPT-2 PC1', 'navy'),
    ('EMBEDDING_2', 'GPT-2 PC2', 'darkblue'),
    ('EMBEDDING_3', 'GPT-2 PC3', 'blue'),
    ('EMBEDDING_4', 'GPT-2 PC4', 'deepskyblue'),
    ('EMBEDDING_5', 'GPT-2 PC5', 'dodgerblue'),
    ('ENTROPY', 'GPT-2 Entropy', 'lightblue')
]

# Define normalization function
def normalize_weights(weights):
    return weights / weights.abs().sum()

# Function to plot anatomical histograms
def plot_anatomical_histograms(df, anatomical_info, features, fig_dir, freq):
    for feature in features:
        plt.figure(figsize=(12, 8))
        sns.histplot(data=df, x=feature, hue=anatomical_info, multiple="stack", palette="viridis")
        plt.title(f'Histogram of {feature} by Anatomical Region ({freq})', fontsize=16, fontweight='bold')
        plt.xlabel('Normalized Weight', fontsize=14, fontweight='bold')
        plt.ylabel('Count', fontsize=14, fontweight='bold')
        plt.legend(title='Anatomical Region')
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f'histogram_{feature}_{freq}.png'))
        plt.show()

# Function to plot anatomical box plots
def plot_anatomical_boxplots(df, anatomical_info, features, fig_dir, freq):
    for feature in features:
        plt.figure(figsize=(12, 8))
        sns.boxplot(x=anatomical_info, y=feature, data=df, palette="viridis")
        plt.title(f'Box Plot of {feature} by Anatomical Region ({freq})', fontsize=16, fontweight='bold')
        plt.xlabel('Anatomical Region', fontsize=14, fontweight='bold')
        plt.ylabel('Normalized Weight', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f'boxplot_{feature}_{freq}.png'))
        plt.show()

# Function to plot anatomical violin plots
def plot_anatomical_violinplots(df, anatomical_info, features, fig_dir, freq):
    for feature in features:
        plt.figure(figsize=(12, 8))
        sns.violinplot(x=anatomical_info, y=feature, data=df, palette="viridis", inner="quartile")
        plt.title(f'Violin Plot of {feature} by Anatomical Region ({freq})', fontsize=16, fontweight='bold')
        plt.xlabel('Anatomical Region', fontsize=14, fontweight='bold')
        plt.ylabel('Normalized Weight', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f'violinplot_{feature}_{freq}.png'))
        plt.show()

# Function to plot anatomical heatmap
def plot_anatomical_heatmap(df, anatomical_info, features, fig_dir, freq):
    summary_df = df.groupby(anatomical_info)[features].mean()  # You can also use sum() or other aggregation functions
    plt.figure(figsize=(14, 10))
    sns.heatmap(summary_df.T, annot=True, cmap='viridis', cbar_kws={'label': 'Average Normalized Weight'})
    plt.title(f'Heatmap of Average Normalized Weights by Anatomical Region ({freq})', fontsize=16, fontweight='bold')
    plt.xlabel('Anatomical Region', fontsize=14, fontweight='bold')
    plt.ylabel('Feature', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f'anatomical_heatmap_{freq}.png'))
    plt.show()

# Main script to read data, normalize, and plot
for freq in freqs:
    feature_data = {}
    available_titles = []
    anatomical_info_df = None
    for input_name, title, color in files_and_titles:
        file_path = f'{csv_dir}/{freq}_CCA_weights_{input_name}.csv'
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['normalized_weight'] = normalize_weights(df['weight'])
            if anatomical_info_df is None:
                anatomical_info_df = df[['ch_name', 'Anat_Label']]
            feature_data[input_name] = df[['ch_name', 'normalized_weight']].set_index('ch_name')
            available_titles.append(title)

    combined_df = pd.concat(feature_data.values(), axis=1)
    combined_df.columns = available_titles
    combined_df = combined_df.merge(anatomical_info_df.set_index('ch_name'), left_index=True, right_index=True)

    # Verify that anatomical_region column exists and is not empty
    anatomical_info = 'Anat_Label'
    if anatomical_info in combined_df.columns and not combined_df[anatomical_info].isnull().all():
        # Plot anatomical histograms
        plot_anatomical_histograms(combined_df, anatomical_info, available_titles, fig_dir, freq)

        # Plot anatomical box plots
        plot_anatomical_boxplots(combined_df, anatomical_info, available_titles, fig_dir, freq)

        # Plot anatomical violin plots
        plot_anatomical_violinplots(combined_df, anatomical_info, available_titles, fig_dir, freq)

        # Plot anatomical heatmap
        plot_anatomical_heatmap(combined_df, anatomical_info, available_titles, fig_dir, freq)
