import pandas as pd

base_path = '/Users/derekrosenzweig/Documents/GitHub/iEEG-Speech'

# Read in the master electrode CSV file
master_electrodes_file = f'{base_path}/scripts/0-electrodes/master-electrodes.csv'
master_electrodes_df = pd.read_csv(master_electrodes_file)

# Get the unique names in the 'name' column
unique_names = master_electrodes_df['sub_name'].unique()
names = master_electrodes_df['sub_name']

# Print the number of unique names
print(f"Number of unique names: {len(unique_names)}")

# Print the list of unique names
# print("Unique names:")
# print(unique_names)
print(len(unique_names)) # 732 unique chan names
print(len(names)) # 1292 total channels across subs

dk = master_electrodes_df['Desikan_Killiany'].unique()
print(f"Number of unique anatomical sites: {len(dk)}")



