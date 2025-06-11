import os
from tqdm import tqdm

from midi import process_midi_separating_instruments_mido
from utils import save_dict

import os
from tqdm import tqdm

from midi import process_midi_separating_instruments_mido
from utils import save_dict, load_pickle_data

OPERATION_TO_DO = [0,1,2]

START = 846
END = 908

MINDIV = 8

SORT_FOLDER = "./To sort"
PROCESSED_FOLDER = "./processed"
MIDI_FOLDERS = ("Fugues", os.path.join("Fugues", "op87"))

REMOVED_TIMING_PATH = "shorten.txt"

OUTPUT_FILE = "data_8.pkl"



# def _get_removed_notes(index):
#     while True:
#         try:
#             removed_notes = input(f"How many beats to remove from BWV_0{index} ? ")
#             if removed_notes.lower() == "r":
#                 index -= 1
#                 continue
#             elif removed_notes.lower() == "e":
#                 return None
#             else:
#                 return index, eval(removed_notes)
#         except:
#             print("Invalid input, please enter a number, 'r' to go to previous file or 'e' to exit.")
#             continue


def _extract_removed_number(file_path:str) -> list:
    removed_numbers = dict()
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split()  # Splitting by spaces
            if parts:  # Ensure there's data
                try:
                    removed_numbers[int((parts[0]))] = eval(parts[1])  # Extract first element as float
                except ValueError:
                    print(f"Error with :'{line}'")
    return removed_numbers


def midi_to_shorten():
    "Midi with prelude before fugue, we removed the not revealing beats."
    global START, END, SORT_FOLDER, PROCESSED_FOLDER, REMOVED_TIMING_PATH, MINDIV
    not_found = []

    if not os.path.exists(PROCESSED_FOLDER):
        os.makedirs(PROCESSED_FOLDER)

    removed_notes = _extract_removed_number(REMOVED_TIMING_PATH)

    for index in tqdm(range(START, END+1), desc=f"Processing midi files...", ncols=75):

        midi_file_path = os.path.join(SORT_FOLDER, f"BWV_0{index}.mid")
        if not os.path.exists(midi_file_path):
            not_found.append(index)
            continue

        data = process_midi_separating_instruments_mido(midi_file_path, MINDIV)
        try:
            for instrument in data.keys():
                if instrument != 'midi_file':
                    data[instrument] = [note for note in data[instrument] if note[0] >= removed_notes[index]] #remove the extra notes
        except Exception as e:
            print(f"Error while processing {midi_file_path} : {e}")
            
                
        save_dict(data, os.path.join(PROCESSED_FOLDER, f"BWV_0{index}.pkl"))


    print("The following files were not found:")
    for i in not_found:
        print(f"BWV_0{i}.mid")

    print("\nAll done!")


def simple_midi():
    "Convert midi in dict of list of (onset, pitch) and save them as `.pkl`."
    global MIDI_FOLDERS, PROCESSED_FOLDER, MINDIV

    for folder in MIDI_FOLDERS:
        files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) and f[-4:] == ".mid"]
        for midifile in tqdm(files, desc=f"Processing midi files of {folder}...", ncols=100):
            data = process_midi_separating_instruments_mido(os.path.join(folder, midifile), MINDIV)
                
                    
            save_dict(data, os.path.join(PROCESSED_FOLDER, midifile[:-4] + ".pkl"))

    print("All done!")


def fusion_all():
    "Fusion all tracks in each file and return all file as one big data `.pkl`."
    global PROCESSED_FOLDER, OUTPUT_FILE
    data = dict()
    files = [f for f in os.listdir(PROCESSED_FOLDER) if os.path.isfile(os.path.join(PROCESSED_FOLDER, f)) and f[-4:] == ".pkl"]
    for file in tqdm(files, desc=f"Merging tracks and files...", ncols=100):
        all_notes = list()
        for key, value in load_pickle_data(os.path.join(PROCESSED_FOLDER, file)).items():
            if key == 'midi_file':
                continue
            all_notes.extend(value)

        data[file[:-4]] = all_notes

    save_dict(data, OUTPUT_FILE)


if __name__ == "__main__":
    operation = (midi_to_shorten, simple_midi, fusion_all)
    for i in OPERATION_TO_DO:
        operation[i]()