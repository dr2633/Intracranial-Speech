import os
import mne
import pandas as pd
from scipy.signal import resample_poly

# Define frequency bands
freq_bands = ['1-40Hz', '70-150Hz']

# Define base paths
base_input_path = '/Users/derekrosenzweig/PycharmProjects/CCA-reduce/derivatives/combined'
base_output_path = '/Users/derekrosenzweig/PycharmProjects/CCA-reduce/derivatives/100-Hz'

# Target sampling rate
target_fs = 100


# Function to process and downsample a single file
def process_file(input_file, output_file):
    # Load the FIF file
    raw = mne.io.read_raw_fif(input_file, preload=True)

    # Get the original sampling rate
    original_fs = raw.info['sfreq']

    # Calculate the downsampling factor
    downsampling_factor = int(original_fs / target_fs)

    # Get the data
    data = raw.get_data()

    # Downsample the data
    data_downsampled = resample_poly(data, up=1, down=downsampling_factor, axis=1)

    # Create a new Raw object with the downsampled data
    info = raw.info.copy()
    info['sfreq'] = target_fs
    raw_downsampled = mne.io.RawArray(data_downsampled, info)

    # Save the downsampled data
    raw_downsampled.save(output_file, overwrite=True)


# Process files for each frequency band
for freq in freq_bands:
    input_dir = os.path.join(base_input_path, freq)
    output_dir = os.path.join(base_output_path, freq)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Loop through all fif files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.fif'):
            input_file = os.path.join(input_dir, filename)
            output_file = os.path.join(output_dir, filename)

            print(f"Processing {input_file}")
            process_file(input_file, output_file)
            print(f"Saved downsampled file to {output_file}")

print("Downsampling complete.")