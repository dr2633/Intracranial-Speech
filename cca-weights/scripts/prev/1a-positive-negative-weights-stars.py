import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

base_path = '/'

# Directory to save figures
fig_dir = f'{base_path}/cca-weights/new_figures'
os.makedirs(fig_dir, exist_ok=True)

# Directory containing the CSV files
csv_dir = f'{base_path}/cca-weights/weights'

# List of frequency bands
freqs = ['70-150Hz']

# List of files and titles
files_and_titles = [
    ('EMBEDDING_1', 'GPT-2 PC1'),
    ('EMBEDDING_2', 'GPT-2 PC2'),
    ('EMBEDDING_3', 'GPT-2 PC3'),
    ('EMBEDDING_4', 'GPT-2 PC4'),
    ('EMBEDDING_5', 'GPT-2 PC5'),
    ('ENTROPY', 'GPT-2 Entropy'),
    ('F0', 'Fundamental Frequency'),
    ('INTENSITY', 'Sound Intensity')
]

def create_dot_plot(df, title, output_path):
    # Calculate absolute weights
    df['abs_weight'] = df['weight'].abs()

    # Determine the top 5% threshold for significance
    threshold = df['abs_weight'].quantile(0.99)
    df['significant'] = df['abs_weight'] >= threshold

    # Separate positive and negative weights
    positive_df = df[df['weight'] > 0]
    negative_df = df[df['weight'] < 0]

    # Sort anatomical sites by the highest absolute weight independently for positive and negative weights
    order_pos = positive_df.groupby('Anat_Label')['abs_weight'].max().sort_values(ascending=False).index
    order_neg = negative_df.groupby('Anat_Label')['abs_weight'].max().sort_values(ascending=False).index

    # Normalize weights for color mapping
    norm_pos = plt.Normalize(positive_df['weight'].min(), positive_df['weight'].max())
    norm_neg = plt.Normalize(negative_df['weight'].min(), negative_df['weight'].max())

    # Create color palettes based on the weights
    palette_pos = sns.color_palette("Blues", as_cmap=True)(norm_pos(positive_df['weight']))
    palette_neg = sns.color_palette("Reds_r", as_cmap=True)(norm_neg(negative_df['weight']))

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 16), sharex=True)

    # Extend x-axis limits to include outliers
    xlim_min = min(df['weight'].min(), -1.1 * df['abs_weight'].max())
    xlim_max = max(df['weight'].max(), 1.1 * df['abs_weight'].max())

    # Plotting positive weights
    sns.stripplot(x='weight', y='Anat_Label', data=positive_df, order=order_pos, palette=palette_pos, ax=axes[0])
    significant_pos = positive_df[positive_df['significant']]
    axes[0].scatter(significant_pos['weight'], significant_pos['Anat_Label'], color='gold', edgecolor='black', s=100, marker='*')
    axes[0].axvline(x=0, color='grey', linestyle='--')
    axes[0].set_title(f'{title} - Positive CCA Weights by Anatomical Region', fontsize=18, fontweight='bold')
    axes[0].set_ylabel('Anatomical Region', fontsize=12)
    axes[0].set_xlim(xlim_min, xlim_max)

    # Update y-axis labels to bold for significant regions
    labels = axes[0].get_yticklabels()
    for label in labels:
        if label.get_text() in significant_pos['Anat_Label'].unique():
            label.set_fontweight('bold')
    axes[0].set_yticklabels(labels)

    # Plotting negative weights
    sns.stripplot(x='weight', y='Anat_Label', data=negative_df, order=order_neg, palette=palette_neg, ax=axes[1])
    significant_neg = negative_df[negative_df['significant']]
    axes[1].scatter(significant_neg['weight'], significant_neg['Anat_Label'], color='gold', edgecolor='black', s=100, marker='*')
    axes[1].axvline(x=0, color='grey', linestyle='--')
    axes[1].set_title(f'{title} - Negative CCA Weights by Anatomical Region', fontsize=18, fontweight='bold')
    axes[1].set_ylabel('Anatomical Region', fontsize=12)
    axes[1].set_xlabel('CCA Weights', fontsize=12)
    axes[1].set_xlim(xlim_min, xlim_max)

    # Update y-axis labels to bold for significant regions
    labels = axes[1].get_yticklabels()
    for label in labels:
        if label.get_text() in significant_neg['Anat_Label'].unique():
            label.set_fontweight('bold')
    axes[1].set_yticklabels(labels)

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
            output_path = os.path.join(fig_dir, f'{freq}_CCA_weights_{input_name}_dot_plot.png')
            create_dot_plot(df, title, output_path)
        else:
            print(f"File not found: {file_path}")
