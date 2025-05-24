import pyxdf
import numpy as np
import os
from pathlib import Path  # Better path handling

# Function to process EEG data from an XDF file
def process_xdf_file(
    xdf_file_path,
    xdf_file_name,
    fs=500,
    n_samples=2000,
    stim_freqs=[7, 8, 9, 11, 7.5, 8.5],
    harmonics=2,
):
    # Load XDF file
    try:
        streams, header = pyxdf.load_xdf(xdf_file_path)
    except Exception as e:
        print(f"Error loading file {xdf_file_path}: {e}")
        return None, None

    # Identify EEG and Event Streams
    eeg_stream = None
    event_stream = None

    for stream in streams:
        if "eeg" in stream["info"]["type"][0].lower():
            eeg_stream = stream
        elif (
            "markers" in stream["info"]["type"][0].lower()
            or "events" in stream["info"]["name"][0].lower()
        ):
            event_stream = stream

    if not eeg_stream or not event_stream:
        print(f"Missing EEG or event stream in {xdf_file_path}. Skipping.")
        return None, None

    # Extract EEG Data
    eeg_data = np.array(eeg_stream["time_series"])  # EEG signals
    eeg_timestamps = np.array(eeg_stream["time_stamps"])  # EEG timestamps

    # Extract Event Markers
    events = [e[0] for e in event_stream["time_series"]]  # Flatten list
    event_timestamps = np.array(event_stream["time_stamps"])

    output_dir = Path(f"./Processing/EdgeSSVEP_Processing/Dataset_txt/{xdf_file_name}/")  # Goes up one level from /scripts
    output_dir.mkdir(parents=True, exist_ok=True)  # Creates all missing parent dirs

    for i in range(1, 25):  # Start1 to Start24
        start_marker = f"Start{i}"
        stop_marker = f"Stop{i}"

        if start_marker in events and stop_marker in events:
            start_idx = events.index(start_marker)
            stop_idx = events.index(stop_marker)

            start_time = event_timestamps[start_idx]
            stop_time = event_timestamps[stop_idx]

            # Extract EEG data within this range
            eeg_indices = np.where(
                (eeg_timestamps >= start_time) & (eeg_timestamps <= stop_time)
            )[0]
            epoch_data = eeg_data[eeg_indices]

            # Save as a text file without a fixed number of timepoints
            txt_file = os.path.join(output_dir, f"trial_{i-1}.txt")
            np.savetxt(txt_file, epoch_data, fmt="%.6f")  # Save with high precision


# Function to process multiple XDF files
def process_multiple_xdf_files(folder_path):
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".xdf"):
            file_path = os.path.join(folder_path, file_name)
            print(f"Processing {file_name}...")
            process_xdf_file(file_path, os.path.splitext(file_name)[0])


    print("Extract txt data from XDF files completed.")



# Example usage
if __name__ == "__main__":
    folder_path = "Datasets"  # Change this to your actual folder path
    process_multiple_xdf_files(folder_path)
