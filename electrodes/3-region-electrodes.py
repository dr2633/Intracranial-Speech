import pandas as pd

base_path = '/Users/derekrosenzweig/Documents/GitHub/iEEG-Speech'

# Read in the master electrode CSV file
master_electrodes_file = f'{base_path}/scripts/0-electrodes/master-electrodes.csv'
master_electrodes_df = pd.read_csv(master_electrodes_file)

# Count the number of electrodes in each anatomical region
electrode_counts = master_electrodes_df['Desikan_Killiany'].value_counts()

# Create a new DataFrame with the region names and electrode counts
region_counts_df = pd.DataFrame({'Region': electrode_counts.index, 'Electrodes': electrode_counts.values})

# Calculate the sum of all electrodes
total_electrodes = region_counts_df['Electrodes'].sum()

# Get the number of unique electrode names
unique_electrodes_count = master_electrodes_df['name'].nunique()

# Save the DataFrame as a CSV file
output_file = 'region_electrode_counts.csv'
region_counts_df.to_csv(output_file, index=False)

print(f"CSV file '{output_file}' has been created with the electrode counts per region.")
print(f"Total number of electrodes: {total_electrodes}")
print(f"Number of electrodes with unique names: {unique_electrodes_count}")
