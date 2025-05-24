import pyxdf
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.cross_decomposition import CCA
import os

# Define Bandpass Filter Function
def bandpass_filter_signal(data, low=2, high=45, order=3, fs=500):
    nyq = 0.5 * fs  # Nyquist frequency
    low_c = low / nyq
    high_c = high / nyq
    b, a = signal.butter(
        order, [low_c, high_c], btype="band"
    )  # Bandpass filter coefficients
    filtered_data = signal.filtfilt(b, a, data, axis=0)  # Apply filter
    return filtered_data


# CCA Reference Signal Generator
def generate_reference_signals(freqs, fs, n_samples, harmonics=3):
    t = np.arange(n_samples) / fs
    reference_signals = {}

    for f in freqs:
        ref = []
        for h in range(1, harmonics + 1):  # Include harmonics
            ref.append(np.sin(2 * np.pi * f * h * t))
            ref.append(np.cos(2 * np.pi * f * h * t))
        reference_signals[f] = np.array(ref)

    return reference_signals


# CCA Algorithm to find best-matching frequency
def apply_cca(eeg_data, reference_signals):
    max_corr = 0
    best_freq = None

    for freq, ref in reference_signals.items():
        cca = CCA(n_components=1)
        # print(eeg_data.shape, ref.shape)  # Debugging line
        cca.fit(eeg_data, ref.T)  # Transpose to match CCA input shape
        X_c, Y_c = cca.transform(eeg_data, ref.T)
        corr = np.corrcoef(X_c[:, 0], Y_c[:, 0])[0, 1]  # Extract correlation

        if corr > max_corr:
            max_corr = corr
            best_freq = freq

    return best_freq, max_corr


# Function to process EEG data from an XDF file
def process_xdf_file(
    xdf_file_path,
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

    # Generate reference signals
    reference_signals = generate_reference_signals(
        stim_freqs, fs, n_samples=n_samples, harmonics=2
    )

    epochs = []
    correct_predictions = 0
    total_trials = 0
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

            # Apply bandpass filter
            filtered_epoch = bandpass_filter_signal(
                epoch_data, low=2, high=45, order=3, fs=fs
            )

            # Take last n_samples samples (handling cases where length < n_samples)
            if filtered_epoch.shape[0] >= n_samples:
                filtered_epoch = filtered_epoch[-n_samples:, :]
            else:
                continue  # Skip if not enough data

            # Assign label using (i-1) % 6 + 1
            label = ((i - 1) % 6) + 1

            # Apply CCA to find the best-matching frequency
            best_freq, max_corr = apply_cca(filtered_epoch, reference_signals)

            # Find the predicted label
            try:
                predicted_label = (
                    stim_freqs.index(best_freq) + 1
                )  # Convert to 1-based index
            except ValueError:
                predicted_label = -1  # If not found

            total_trials += 1

            # Check correctness
            if predicted_label == label:
                correct_predictions += 1

            # Store results
            epochs.append(
                {
                    "File": os.path.basename(xdf_file_path),
                    "Start": start_time,
                    "Stop": stop_time,
                    "Label": label,
                    "CCA_Label": predicted_label,
                    "Best_Freq": best_freq,
                    "Correlation": max_corr,
                }
            )

    # Compute accuracy
    accuracy = correct_predictions / total_trials
    return epochs, accuracy


# Function to process multiple XDF files
def process_multiple_xdf_files(folder_path):
    summary = []
    all_epochs = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".xdf"):
            file_path = os.path.join(folder_path, file_name)
            print(f"Processing {file_name}...")

            epochs, accuracy = process_xdf_file(file_path)

            if epochs is not None:
                all_epochs.extend(epochs)
                summary.append({"File": file_name, "Accuracy": accuracy * 100})

    # Save full results to CSV
    epoch_df = pd.DataFrame(all_epochs)
    epoch_df.to_csv("Results/full_cca_results.csv", index=False)
    print("Full results saved to Results/full_cca_results.csv")

    # Save summary file
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv("Results/summary_cca_results.csv", index=False)
    print("Summary results saved to Results/summary_cca_results.csv")

    print(summary_df)


# Example usage
if __name__ == "__main__":
    folder_path = "Datasets"  # Change this to your actual folder path
    process_multiple_xdf_files(folder_path)
