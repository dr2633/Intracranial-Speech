import pandas as pd

# Parameters
sub = 'sub-08'
ses = 'ses-01'
s_num = 'S24_224_TN'

base_path = '/Users/derekrosenzweig/Documents/GitHub/iEEG-Speech'

def excel_to_tsv(input_excel_file, output_tsv_file):
    try:
        # Read the Excel file into a pandas DataFrame
        df = pd.read_excel(input_excel_file)

        # Save the DataFrame to a TSV file
        df.to_csv(output_tsv_file, sep='\t', index=False)

        print(f'Successfully converted {input_excel_file} to {output_tsv_file}')
    except Exception as e:
        print(f'Error: {str(e)}')

if __name__ == "__main__":
    input_excel_file = f'{base_path}/BIDS/{sub}/{ses}/ieeg/{s_num}_elec_loc.xlsx'
    output_tsv_file = f"{base_path}/BIDS/{sub}/{ses}/ieeg/{sub}_{ses}_electrodes.tsv"  # Replace with the desired output TSV file path

    excel_to_tsv(input_excel_file, output_tsv_file)