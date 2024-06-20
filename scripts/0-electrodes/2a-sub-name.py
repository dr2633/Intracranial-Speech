import pandas as pd

elec = '/Users/derekrosenzweig/Documents/GitHub/iEEG-Speech/scripts/0-electrodes/master-electrodes.csv'

df = pd.read_csv(elec)

# Combine 'sub' and 'name' columns
df['sub_name'] = df.apply(lambda row: f"{row['sub']} {row['name']}", axis=1)

# Optionally, if you want to drop the original 'sub' and 'name' columns:
# df.drop(columns=['sub', 'name'], inplace=True)

print(df)
