import os

from midi import process_midi_separating_instruments_mido
from utils import save_dict

START = 846
END = 909
SORT_FOLDER = "./To sort"
PROCESSED_FOLDER = "./processed"

not_found = []

if not os.path.exists(PROCESSED_FOLDER):
    os.makedirs(PROCESSED_FOLDER)


def _get_removed_notes(index):
    while True:
        try:
            removed_notes = input(f"How many beats to remove from BWV_0{index} ? ")
            if removed_notes.lower() == "r":
                index -= 1
                continue
            elif removed_notes.lower() == "e":
                return None
            else:
                return index, eval(removed_notes)
        except:
            print("Invalid input, please enter a number, 'r' to go to previous file or 'e' to exit.")
            continue

index = START
while index <= END:
    out = _get_removed_notes(index)
    if out is None:
        print(f"End of process at BWV_0{index}.mid")
        break
    index, removed_notes = out
    
        
    midi_file_path = os.path.join(SORT_FOLDER, f"BWV_0{index}.mid")
    if not os.path.exists(midi_file_path):
        not_found.append(index)
        continue

    data = process_midi_separating_instruments_mido(midi_file_path)
    for instrument in data.keys():
        if instrument != 'midi_file':
            data[instrument] = [note for note in data[instrument] if note[0] >= removed_notes] #remove the extra notes
            
    save_dict(data, os.path.join(PROCESSED_FOLDER, f"BWV_0{index}.pkl"))

    index += 1

print("The following files were not found:")
for i in not_found:
    print(f"BWV_0{i}.mid")

print("\nAll done!")