import pandas as pd
import numpy as np
from scipy.spatial import distance

feature = 'ENTROPY'


# Read CSV file
file = f'/Users/derekrosenzweig/PycharmProjects/CCA-reduce/cca-weights/weights/70-150Hz_CCA_weights_{feature}.csv'
df = pd.read_csv(file)

# Filter for WM and GM
wm_sites = df[df['WMvsGM'] == 'WM']
gm_sites = df[df['WMvsGM'] == 'GM']


# Create a function to compute the Euclidean distance
def compute_euclidean_distance(coord1, coord2):
    return np.linalg.norm(coord1 - coord2)


# Create lists to store results
nearest_neighbors = []
distances = []

# Iterate over each WM site
for index_wm, wm_row in wm_sites.iterrows():
    wm_coord = np.array(
        [wm_row['fsaverageINF_coord_1'], wm_row['fsaverageINF_coord_2'], wm_row['fsaverageINF_coord_3']])

    # Initialize minimum distance and nearest GM site
    min_distance = float('inf')
    nearest_gm_site = None

    # Iterate over each GM site to find the nearest neighbor
    for index_gm, gm_row in gm_sites.iterrows():
        gm_coord = np.array(
            [gm_row['fsaverageINF_coord_1'], gm_row['fsaverageINF_coord_2'], gm_row['fsaverageINF_coord_3']])
        dist = compute_euclidean_distance(wm_coord, gm_coord)

        if dist < min_distance:
            min_distance = dist
            nearest_gm_site = gm_row['Anat']

    # Store the results
    nearest_neighbors.append(nearest_gm_site)
    distances.append(min_distance)

# Add the results to the DataFrame
wm_sites['Nearest_GM_Site'] = nearest_neighbors
wm_sites['Distance_to_GM'] = distances

# Print the updated DataFrame with nearest GM sites and distances
print(wm_sites[['Anat', 'fsaverageINF_coord_1', 'fsaverageINF_coord_2', 'fsaverageINF_coord_3', 'Nearest_GM_Site',
                'Distance_to_GM']])

# Optionally, save the results to a new CSV file
output_file = f'/Users/derekrosenzweig/PycharmProjects/CCA-reduce/cca-weights/nearest-neighbors/{feature}_nearest_gm_sites.csv'
wm_sites.to_csv(output_file, index=False)
