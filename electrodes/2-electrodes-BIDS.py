# Step 2: Make electrodes TSV BIDS compatible
# Add required column names and store TSV in BIDS BOX folder

import pandas as pd
import os

# Parameters
sub = "sub-08"
ses = "ses-01"

user_paths = ['/Users/derekrosenzweig/Documents/GitHub/iEEG-Speech']

# set base path depending on who is running the code
for user_path in user_paths:
    if os.path.exists(user_path):
        base_path = user_path



elec_path = f'{base_path}/BIDS/{sub}/{ses}/ieeg/{sub}_{ses}_electrodes.tsv'

# Read the existing TSV file into a DataFrame
df = pd.read_csv(elec_path, sep='\t')

# Create the new columns based on your requirements
df['name'] = df['FS_label']
df['x'] = df['MNI_coord_1']
df['y'] = df['MNI_coord_2']
df['z'] = df['MNI_coord_3']
df['size'] = '3.5 mm'

# Specify the column order
column_order = ['name', 'x', 'y', 'z', 'size'] + [col for col in df.columns if col not in ['name', 'x', 'y', 'z', 'size']]

# Reorder the columns
df = df[column_order]

# Save the updated DataFrame back to the TSV file
df.to_csv(elec_path, sep='\t', index=False)