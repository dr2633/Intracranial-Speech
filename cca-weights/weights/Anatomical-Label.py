import os
import pandas as pd

# Path configuration
base_path = '/Users/derekrosenzweig/PycharmProjects/CCA-reduce'
csv_dir = os.path.join(base_path, 'cca-weights', 'weights')

# List of files
files_and_titles = [
    'F0',
    'INTENSITY',
    'EMBEDDING_1',
    'EMBEDDING_2',
    'EMBEDDING_3',
    'EMBEDDING_4',
    'EMBEDDING_5',
    'ENTROPY',
    '5-PCs'
]

# Consolidation and abbreviation mapping
consolidation_mapping = {
    'Banks of Superior Temporal Sulcus (STS)': 'STG',
    'Caudal Anterior Cingulate Cortex': 'cACC',
    'Caudal Middle Frontal Gyrus': 'MFG',
    'Entorhinal Cortex': 'Entorhinal',
    'Fusiform Gyrus': 'Fusiform',
    'Heschl\'s Gyrus (HG)': 'HG',
    'Inferior Frontal Gyrus (IFG)': 'IFG',
    'Inferior Parietal Lobule': 'IPL',
    'Inferior Temporal Gyrus (ITG)': 'ITG',
    'Insula': 'Insula',
    'Isthmus Cingulate Cortex': 'ICC',
    'Lateral Orbitofrontal Cortex': 'OFC',
    'Lingual Gyrus': 'Lingual',
    'Medial Orbitofrontal Cortex': 'OFC',
    'Middle Temporal Gyrus (MTG)': 'MTG',
    'Paracentral Lobule': 'Paracentral',
    'Parahippocampal Gyrus': 'PHG',
    'Postcentral Gyrus': 'Postcentral',
    'Posterior Cingulate Cortex': 'PCC',
    'Precentral Gyrus': 'Precentral',
    'Precuneus': 'Precuneus',
    'Rostral Anterior Cingulate Cortex': 'rACC',
    'Rostral Middle Frontal Gyrus': 'MFG',
    'Superior Frontal Gyrus': 'SFG',
    'Superior Parietal Lobule': 'SPL',
    'Superior Temporal Gyrus (STG)': 'STG',
    'Supramarginal Gyrus': 'SMG',
    'Temporal Pole': 'TP'
}

# Iterate through the list of files and apply abbreviations and consolidations
for analysis_type in files_and_titles:
    file_path = os.path.join(csv_dir, f'70-150Hz_CCA_weights_{analysis_type}.csv')

    try:
        # Load the data
        data = pd.read_csv(file_path)

        # Apply consolidation and abbreviation
        data['Anatomical-Final'] = data['Anat_Label'].map(consolidation_mapping).fillna(data['Anat_Label'])

        # Save the updated data
        data.to_csv(file_path, index=False)

        print(f"Abbreviated file saved: {file_path}")

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

print("Finished abbreviating and consolidating anatomical labels.")




# import os
# import pandas as pd
#
# # Path configuration
# base_path = '/Users/derekrosenzweig/PycharmProjects/CCA-reduce'
# csv_dir = os.path.join(base_path, 'cca-weights', 'weights')
#
# # List of files
# files_and_titles = [
#     'F0',
#     'INTENSITY',
#     'EMBEDDING_1',
#     'EMBEDDING_2',
#     'EMBEDDING_3',
#     'EMBEDDING_4',
#     'EMBEDDING_5',
#     'ENTROPY'
# ]
#
# # Dictionary to hold anatomical labels and their counts
# anatomical_labels_counts = {}
#
# # Iterate through the list of files
# for analysis_type in files_and_titles:
#     file_path = os.path.join(csv_dir, f'70-150Hz_CCA_weights_{analysis_type}.csv')
#
#     try:
#         # Load the data
#         data = pd.read_csv(file_path)
#
#         # Count occurrences of each anatomical label
#         anat_label_counts = data['Anat_Label'].value_counts().to_dict()
#
#         # Update the dictionary with counts
#         for label, count in anat_label_counts.items():
#             if label in anatomical_labels_counts:
#                 anatomical_labels_counts[label] += count
#             else:
#                 anatomical_labels_counts[label] = count
#
#     except Exception as e:
#         print(f"Error processing file {file_path}: {e}")
#
# # Print all unique anatomical labels and their counts
# print("Anatomical Labels and their Counts:")
# for label, count in sorted(anatomical_labels_counts.items()):
#     print(f"{label}: {count}")
