import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

# Mapping dictionary for consolidating and abbreviating anatomical labels
anatomy_mapping = {
    'Medial Orbitofrontal Cortex': 'OFC',
    'Lateral Orbitofrontal Cortex': 'OFC',
    'Inferior Frontal Gyrus (IFG)': 'IFG',
    'Caudal Anterior Cingulate Cortex': 'ACC',
    'Rostral Anterior Cingulate Cortex': 'ACC',
    'Superior Temporal Gyrus (STG)': 'STG',
    'Rostral Middle Frontal Gyrus': 'MFG',
    'Supramarginal Gyrus': 'SMG',
    'Parahippocampal Gyrus': 'PHG',
    'Posterior Cingulate Cortex': 'PCC',
    'Insula': 'INS',
    'Fusiform Gyrus': 'FG',
    'Entorhinal Cortex': 'EC',
    'Lingual Gyrus': 'LG',
    'Precentral Gyrus': 'PreCG',
    'Superior Frontal Gyrus': 'SFG',
    'Paracentral Lobule': 'PCL',
    'Middle Temporal Gyrus (MTG)': 'MTG',
    'Banks of Superior Temporal Sulcus (STS)': 'STS',
    'Inferior Temporal Gyrus (ITG)': 'ITG',
    "Heschl's Gyrus (HG)": 'HG',
    'Caudal Middle Frontal Gyrus': 'MFG',
    'Superior Parietal Lobule': 'Parietal',
    'Precuneus': 'Parietal',
    'Inferior Parietal Lobule': 'Parietal',
    'Temporal Pole': 'TP',
    'Isthmus Cingulate Cortex': 'ICC',
    'Postcentral Gyrus': 'PostCG'
}

# Function to plot anatomical heatmap using maximum absolute values of normalized weights
def plot_anatomical_heatmap(df, anatomical_info, features, fig_dir, freq):
    # Take the absolute values of the normalized weights
    abs_df = df[features].abs()
    summary_df = abs_df.groupby(df[anatomical_info]).max()  # Using max() for the highest contribution
    plt.figure(figsize=(14, 10))
    sns.heatmap(summary_df.T, annot=True, cmap='viridis', cbar_kws={'label': 'Maximum Absolute Normalized Weight'})
    plt.title(f'Heatmap of Maximum Absolute Normalized Weights by Anatomical Region ({freq})', fontsize=18, fontweight='bold')
    plt.xlabel('Anatomical Region', fontsize=14, fontweight='bold')
    plt.ylabel('Feature', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f'anatomical_heatmap_{freq}_max.png'))
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

    # Consolidate and abbreviate anatomical labels
    combined_df['Anatomy'] = combined_df['Anat_Label'].map(anatomy_mapping)

    # Verify that 'Anatomy' column exists and is not empty
    anatomical_info = 'Anatomy'
    if anatomical_info in combined_df.columns and not combined_df[anatomical_info].isnull().all():
        # Plot anatomical heatmap
        plot_anatomical_heatmap(combined_df, anatomical_info, available_titles, fig_dir, freq)
