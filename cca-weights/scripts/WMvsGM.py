import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from statsmodels.stats.proportion import proportions_ztest

# Path configuration
base_path = '/Users/derekrosenzweig/PycharmProjects/CCA-reduce'
csv_dir = os.path.join(base_path, 'cca-weights', 'weights')
fig_dir = os.path.join(base_path, 'figures', 'WMvsGM')
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

# Function to create individual barplots
def create_barplot(data, analysis_type, title, color, output_path, max_y):
    # Get the percentage of significant electrodes for WM and GM
    wm_gm_counts = data['WMvsGM'].value_counts()
    total_significant = data.shape[0]
    percentages = (wm_gm_counts / total_significant) * 100

    plt.figure(figsize=(8, 6))
    ax = percentages.plot(kind='bar', color=['darkgrey', 'lightgrey'])

    # Customize plot appearance
    plt.title(f'{title} - WM vs GM Distribution', fontsize=18, fontweight='bold')
    plt.ylabel('Percentage of Significant Electrodes', fontsize=14, fontweight='bold')
    plt.ylim(0, max_y * 1.1)  # Add buffer room on the y-axis
    plt.xticks(rotation=0, ha='center', fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Remove spines for a cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Remove the x-axis label
    ax.set_xlabel('')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# Load all electrodes data to calculate overall WM and GM percentages
all_electrodes_data = []
for analysis_type, title, color in files_and_titles:
    file_path = os.path.join(csv_dir, f'70-150Hz_CCA_weights_{analysis_type}.csv')
    try:
        data = pd.read_csv(file_path)
        all_electrodes_data.append(data)
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

# Concatenate all data to get overall WM and GM percentages
all_electrodes_data = pd.concat(all_electrodes_data, ignore_index=True)

# Calculate overall WM and GM percentages
overall_wm_gm_counts = all_electrodes_data['WMvsGM'].value_counts()
total_electrodes = all_electrodes_data.shape[0]
overall_percentages = (overall_wm_gm_counts / total_electrodes) * 100

print("Overall WM vs GM percentages:")
print(overall_percentages)

# Determine the maximum y-axis limit across all features
max_y = 0
wm_gm_data = []
p_values = []
for analysis_type, title, color in files_and_titles:
    file_path = os.path.join(csv_dir, f'70-150Hz_CCA_weights_{analysis_type}.csv')

    try:
        # Load the data
        data = pd.read_csv(file_path)

        # Filter data to include only significant electrodes
        z_scores = stats.zscore(data['weight'])
        data['z_score'] = z_scores
        significant_data = data[np.abs(data['z_score']) > z_threshold]

        # Get the percentage of significant electrodes for WM and GM
        wm_gm_counts = significant_data['WMvsGM'].value_counts()
        total_significant = significant_data.shape[0]
        percentages = (wm_gm_counts / total_significant) * 100

        # Perform a two-proportion z-test
        wm_count = wm_gm_counts.get('WM', 0)
        overall_wm_count = overall_wm_gm_counts.get('WM', 0)
        stat, p_value = proportions_ztest([wm_count, overall_wm_count], [total_significant, total_electrodes])
        p_values.append(p_value)

        # Update max_y if the current plot's max percentage is higher
        max_y = max(max_y, percentages.max())

        # Collect data for the CSV file
        wm_gm_data.append((title, total_significant, percentages.get('GM', 0), percentages.get('WM', 0), p_value))

        # Create and save the barplot
        output_path = os.path.join(fig_dir, f'{analysis_type}_WMvsGM_barplot.png')
        create_barplot(significant_data, analysis_type, title, color, output_path, max_y)

        print(f"Barplot saved: {output_path}")

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

# Create combined barplot
wm_gm_df = pd.DataFrame(wm_gm_data, columns=['Feature', 'Total_Significant', 'GM_Percentage', 'WM_Percentage', 'p_value'])
wm_gm_df.set_index('Feature', inplace=True)
ax = wm_gm_df[['GM_Percentage', 'WM_Percentage']].plot(kind='bar', figsize=(14, 8), color=['darkgrey', 'lightgrey'])

# Customize plot appearance
plt.title('WM vs GM Distribution Across All Features', fontsize=22, fontweight='bold')
plt.ylabel('Percentage of Significant Electrodes', fontsize=16, fontweight='bold')
plt.ylim(0, max_y * 1.1)  # Add buffer room on the y-axis
plt.axhline(y=overall_percentages.get('WM', 0), color='black', linestyle='--', label='WM Percentage for All Electrodes')  # Add dashed line
plt.xticks(rotation=45, ha='right', fontsize=14, fontweight='bold')
plt.yticks(fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(['GM', 'WM', 'WM Percentage for All Electrodes'], fontsize=14)

# Remove spines for a cleaner look
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.tight_layout()
combined_output_path = os.path.join(fig_dir, 'WMvsGM_combined_barplot.png')
plt.savefig(combined_output_path)
plt.close()

print(f"Combined barplot saved: {combined_output_path}")

# Save the DataFrame as a CSV file
stats_csv_path = os.path.join(fig_dir, 'WMvsGM_statistics.csv')
wm_gm_df.to_csv(stats_csv_path)
print(f"Statistics saved to {stats_csv_path}")

# Create methods description
methods_description = f"""
We analyzed the distribution of white matter (WM) and grey matter (GM) electrodes across various feature sets. 
We computed the percentage of WM and GM electrodes within significant clusters for each feature set using a Z-score threshold of {z_threshold}.
The overall percentages of WM and GM electrodes across all electrodes were used as benchmarks. 
A two-proportion z-test was performed to compare the WM percentages of significant electrodes for each feature set to the overall WM percentage.

The overall distribution of electrodes was as follows:
- Total number of electrodes: {total_electrodes}
- Percentage of GM electrodes: {overall_percentages.get('GM', 0):.2f}%
- Percentage of WM electrodes: {overall_percentages.get('WM', 0):.2f}%

For each feature set, the following statistics were computed and saved in {stats_csv_path}:
"""

for index, row in wm_gm_df.iterrows():
    if pd.isna(row['p_value']):
        p_value_str = 'N/A'
    else:
        p_value_str = f"{row['p_value']:.4f}"

    methods_description += f"""
    Feature: {index}
    - Total significant electrodes: {int(row['Total_Significant'])}
    - Percentage of GM electrodes: {row['GM_Percentage']:.2f}%
    - Percentage of WM electrodes: {row['WM_Percentage']:.2f}%
    - p-value (WM vs overall WM): {p_value_str}
    """

methods_file_path = os.path.join(fig_dir, 'methods_description.txt')
with open(methods_file_path, 'w') as f:
    f.write(methods_description)

print(f"Methods description saved to {methods_file_path}")
print("Finished creating barplots and statistics for WM vs GM distribution.")
