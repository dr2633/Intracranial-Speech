BIDS_path = '/Users/derekrosenzweig/Documents/GitHub/iEEG-Speech/BIDS'

elec_path = f'{BIDS_path}/{sub}/{ses}/{sub}_{ses}_electrodes.tsv'


# Create a for loop that gets files for sub = 'sub-01' through 'sub-07'
for i in range(2, 9):  # Iterate through sub-01 to sub-07
    if i == 4:
        continue


