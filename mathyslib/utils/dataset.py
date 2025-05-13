import os
import pandas as pd
from fractions import Fraction

from mathyslib.utils.midi import process_midi_separating_instruments


def preprocess_dataset_to_csv(folder_path, output_csv_path):
    """
    Preprocesses the dataset by extracting pitch, onset, and duration from MIDI files,
    and saves the information to a CSV file.

    Args:
      folder_path: Path to the folder containing MIDI files.
      output_csv_path: Path to the output CSV file.
    """
    all_data = []

    # Create output folder if it doesn't exist
    output_folder = os.path.join(os.path.dirname(output_csv_path), 'tracks_csv')
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(folder_path):
        if filename.endswith(".mid") or filename.endswith(".midi"):
            midi_file_path = os.path.join(folder_path, filename)
            try:
                midi_data = process_midi_separating_instruments(midi_file_path)
                all_data.extend(midi_data)

                # Save a separate CSV for each track
                track_csv_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.csv")
                pd.DataFrame(midi_data).to_csv(track_csv_path, index=False)
                print(f"Track CSV saved to {track_csv_path}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue

    # Save full dataset to CSV
    df = pd.DataFrame(all_data)
    df.to_csv(output_csv_path, index=False)
    print(f"Dataset saved to {output_csv_path}")
# # Usage
# output_csv_path = "/content/drive/MyDrive/IRMA/dataset.csv"
# preprocess_dataset_to_csv(folder_path, output_csv_path)


def load_preprocessed_dataset_from_csv(csv_folder_path):
    """
    Loads preprocessed CSV files and reconstructs the dataset.

    Args:
      csv_folder_path: Path to the folder containing track CSV files.

    Returns:
      A dictionary where keys are MIDI file names and values are dictionaries of
      track and instrument data (pitch and onset of each notes).
    """
    dataset = {}

    for filename in os.listdir(csv_folder_path):
        if filename.endswith(".csv"):
            csv_path = os.path.join(csv_folder_path, filename)
            try:
                df = pd.read_csv(csv_path)
                midi_file = df['midi_file'].iloc[0]

                if midi_file not in dataset:
                    dataset[midi_file] = {}

                for instrument, group in df.groupby('instrument'):
                    if instrument not in dataset[midi_file]:
                        dataset[midi_file][instrument] = []

                    for row in group.itertuples(index=False):
                        onset = Fraction(row.onset) if '/' in str(row.onset) else float(row.onset)
                        dataset[midi_file][instrument].append((onset, row.pitch))
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue

    return dataset


def count_onset_apparition(preprocessed_dataset):
    """Extracts fractional values from onset points and returns a dictionary with their counts."""
    decimal_counts = {}

    for track_data in preprocessed_dataset.values():
        for instrument_data in track_data.values():
            for onset, _ in instrument_data:
                decimal_part = (onset % 1)  # Ensure precision for small decimals
                if decimal_part not in decimal_counts:
                    decimal_counts[decimal_part] = 0
                decimal_counts[decimal_part] += 1

    return decimal_counts


def get_subsets_with_points(preprocessed_dataset):
    """
    Generates subsets in the format '<track_filename>_<instrument_name>' with corresponding points,
    excluding manually specified tracks.

    Args:
        preprocessed_dataset: The preprocessed dataset dictionary.

    Returns:
        A dictionary where keys are subsets ('<track_filename>_<instrument_name>') and values are lists of (onset, pitch) points.
    """

    subsets_data = {}
    for track_name, instruments_data in preprocessed_dataset.items():
        # Extract only the filename part
        track_filename = track_name.split(" - ")[-1]

        for instrument_name, points in instruments_data.items():
            subset_name = f"{track_filename}_{instrument_name}"  # Using cleaned track name
            subsets_data[subset_name] = points

    return subsets_data