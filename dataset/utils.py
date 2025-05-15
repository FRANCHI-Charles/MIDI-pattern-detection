import os
import os.path
import pickle
from tqdm import tqdm

from dataset.midi import process_midi_separating_instruments


def load_pop909(folderpath:str, mindiv:int=24) -> dict:
    """
    Load the pop909 dataset as a dict with track ID as key and the output
    of `process_midi_separating_instruments` as value, that is to say:
    a dict of instrument name as key and a list of notes (onset, pitch, duration) as value)
    """
    data = dict()
    for subfolder in tqdm(os.listdir(folderpath), desc=f"Loading midi files...", ncols=75):
        full_subfolder_path = os.path.join(folderpath, subfolder)
        if os.path.isdir(full_subfolder_path):
            try:
                data[subfolder] = process_midi_separating_instruments(os.path.join(full_subfolder_path, subfolder + ".mid"), os.path.join(full_subfolder_path, "beat_midi.txt"), mindiv)
            except Exception as e:
                print(f"Error while processing {subfolder} : {e}")
    return data


### Save and load functions

def save_dict(data:dict, filename:str):
    """Save a dict as pickle."""
    with open(filename, "wb") as file:
        pickle.dump(data, file)


def load_pickle_data(filename:str):
    with open(filename, "rb") as file :
        return pickle.load(file)