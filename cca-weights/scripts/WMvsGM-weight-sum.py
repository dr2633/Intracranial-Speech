import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from statsmodels.stats.proportion import proportions_ztest
from itertools import combinations

# Path configuration
base_path = '/Users/derekrosenzweig/PycharmProjects/CCA-reduce'
csv_dir = os.path.join(base_path, 'cca-weights', 'weights')
fig_dir = os.path.join(base_path, 'figures', 'WMvsGM1')
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

# Calculate counts of WM and GM electrodes
wm_count_all = all_electrodes_data[all_electrodes_data['WMvsGM'] == 'WM'].shape[0]
gm_count_all = all_electrodes_data[all_electrodes_data['WMvsGM'] == 'GM'].shape[0]
total_count_all = all_electrodes_data.shape[0]

overall_wm_percentage = (wm_count_all / total_count_all) * 100
overall_gm_percentage = (gm_count_all / total_count_all) * 100

# Calculate the total number of unique electrodes
total_electrodes = all_electrodes_data['sub_name'].nunique()

print("Overall WM vs GM percentages based on counts:")
print(f"GM: {overall_gm_percentage:.2f}%")
print(f"WM: {overall_wm_percentage:.2f}%")
print(f"Total number of electrodes: {total_electrodes}")

# Determine the maximum y-axis limit across all features
max_y = 0
wm_gm_data = []
p_values_ztest = []
for analysis_type, title, color in files_and_titles:
    file_path = os.path.join(csv_dir, f'70-150Hz_CCA_weights_{analysis_type}.csv')

    try:
        # Load the data
        data = pd.read_csv(file_path)

        # Filter data to include only significant electrodes
        z_scores = stats.zscore(data['weight'])
        data['z_score'] = z_scores
        significant_data = data[np.abs(data['z_score']) > z_threshold]

        # Calculate counts of WM and GM electrodes
        wm_count = significant_data[significant_data['WMvsGM'] == 'WM'].shape[0]
        gm_count = significant_data[significant_data['WMvsGM'] == 'GM'].shape[0]
        total_count = significant_data.shape[0]

        wm_percentage = (wm_count / total_count) * 100
        gm_percentage = (gm_count / total_count) * 100

        # Perform a two-proportion z-test
        count = np.array([wm_count, wm_count_all])
        nobs = np.array([total_count, total_count_all])
        stat, p_value = proportions_ztest(count, nobs)
        p_values_ztest.append(p_value)

        # Update max_y if the current plot's max percentage is higher
        max_y = max(max_y, wm_percentage, gm_percentage)

        # Collect data for the CSV file
        wm_gm_data.append((title, total_count, gm_percentage, wm_percentage, p_value))

        # Create and save the barplot
        output_path = os.path.join(fig_dir, f'{analysis_type}_WMvsGM_barplot.png')
        create_barplot(significant_data, analysis_type, title, color, output_path, max_y)

        print(f"Barplot saved: {output_path}")

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

# Create combined barplot
wm_gm_df = pd.DataFrame(wm_gm_data, columns=['Feature', 'Total_Significant_Count', 'GM_Percentage', 'WM_Percentage', 'P_Value'])
wm_gm_df.set_index('Feature', inplace=True)
ax = wm_gm_df[['GM_Percentage', 'WM_Percentage']].plot(kind='bar', figsize=(14, 8), color=['darkgrey', 'lightgrey'])

# Customize plot appearance
plt.title('WM vs GM Percentage Distribution Across All Features', fontsize=22, fontweight='bold')
plt.ylabel('Percentage', fontsize=16, fontweight='bold')
plt.ylim(0, max_y * 1.1)  # Add buffer room on the y-axis
plt.axhline(y=overall_wm_percentage, color='black', linestyle='--', label='WM Percentage for All Electrodes')  # Add dashed line
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

# Perform chi-square tests between distributions
chi_square_tests = []
for (f1, title1, _), (f2, title2, _) in combinations(files_and_titles, 2):
    data1 = wm_gm_df.loc[title1]
    data2 = wm_gm_df.loc[title2]
    observed = np.array([[data1['GM_Percentage'], data1['WM_Percentage']],
                         [data2['GM_Percentage'], data2['WM_Percentage']]])
    chi2, p_value, _, _ = stats.chi2_contingency(observed)
    chi_square_tests.append((title1, title2, chi2, p_value))

# Save the chi-square test results
chi_square_tests_df = pd.DataFrame(chi_square_tests, columns=['Feature1', 'Feature2', 'chi2_stat', 'p_value'])
chi_square_tests_csv_path = os.path.join(fig_dir, 'chi_square_tests_statistics.csv')
chi_square_tests_df.to_csv(chi_square_tests_csv_path)
print(f"Chi-square test statistics saved to {chi_square_tests_csv_path}")

# Create methods description
methods_description = f"""
We analyzed the distribution of white matter (WM) and grey matter (GM) electrodes across various feature sets. 
We computed the percentage of WM and GM electrodes within significant clusters for each feature set using a Z-score threshold of {z_threshold}.
The overall percentages of WM and GM electrodes across all electrodes were used as benchmarks.

The overall distribution of electrodes was as follows:
- Total number of electrodes: {total_electrodes}
- GM percentage: {overall_gm_percentage:.2f}%
- WM percentage: {overall_wm_percentage:.2f}%

For each feature set, the following statistics were computed and saved in {stats_csv_path}:
"""

for index, row in wm_gm_df.iterrows():
    methods_description += f"""
    Feature: {index}
    - Total significant count: {row['Total_Significant_Count']}
    - GM percentage: {row['GM_Percentage']:.2f}%
    - WM percentage: {row['WM_Percentage']:.2f}%
    - p-value: {row['P_Value']:.4f}
    """

methods_description += f"\n\nAdditionally, chi-square tests were performed to compare the GM and WM percentages across different feature sets. The results are saved in {chi_square_tests_csv_path}."

methods_file_path = os.path.join(fig_dir, 'methods_description.txt')
with open(methods_file_path, 'w') as f:
    f.write(methods_description)

print(f"Methods description saved to {methods_file_path}")
print("Finished creating barplots and statistics for WM vs GM distribution.")
