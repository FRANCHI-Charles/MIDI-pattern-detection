import mido
import pretty_midi
import numpy as np
from fractions import Fraction
import os

def _msgs_to_notes(track):
    "return a list of notes as (onset_in_ticks_since_start, pitch)"
    notes =list()
    time = 0
    for msg in track:
        time += msg.time
        if msg.type == 'note_on' and msg.velocity > 0:
            notes.append([time, msg.note])
    return notes

def _extract_beat_times(file_path:str) -> list:
    float_numbers = [0]
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split()  # Splitting by spaces
            if parts:  # Ensure there's data
                try:
                    float_numbers.append(float(parts[0]))  # Extract first element as float
                except ValueError:
                    print(f"Warning: a first element in '{file_path}' is not a float : '{line}'")
    return float_numbers


def _find_nearest_division(time:float, beat_times:list, mindiv:int) -> Fraction:
    beat_times = sorted(beat_times)
    previous_beat = sum([beat <= time for beat in beat_times]) - 1
    if previous_beat <= -1:
        raise ValueError(f"time smaller than the smallest element of beat_times : {time} < {beat_times[0]}")
    if previous_beat == len(beat_times) - 1:
        beat_times.append(2*beat_times[-1] - beat_times[-2]) # approximation of the missing last "beat"
        if  time > beat_times[-1]:
            raise ValueError(f"At least two beats are missing to find the position of `time` : {time} > {beat_times[-2]}")
        
    previous_beat_time = beat_times[previous_beat]
    next_beat_time = beat_times[previous_beat + 1]
    
    scale = np.linspace(previous_beat_time, next_beat_time, mindiv+1)
    nearest_div = (np.abs(scale - time)).argmin()
    
    return previous_beat + Fraction(nearest_div, mindiv)


def process_midi_separating_instruments_mido(midi_file_path:str, mindiv:int=24) -> dict:
    """Processes a MIDI file and returns note information (onset, pitch)
    separated by instruments with onsets normalized to beats.

    Args:
      midi_file_path: Path to the MIDI file.
      mindiv:int, 1/tatum (default is 24 to handle double semiquaver and triplet)
      return_duration: bool, If True, returns the duration of the notes in beats.

    Returns:
      A dictionaries with keys 'midi_file' with the file name, and one key per 'instrument' with a list of values [note onset, note pitch].
    """
    
    midi_data = mido.MidiFile(midi_file_path)

    

    data = dict()
    data['midi_file'] = os.path.basename(midi_file_path)

    for i, track in enumerate(midi_data.tracks):
        instrument_name = f'Track {i}'
        notes = _msgs_to_notes(track)
        if len(notes) == 0:
            continue

        data[instrument_name] = list()

        for note in notes:
            # Convert ticks to note per beat
            note[0] = (note[0] * mindiv) / midi_data.ticks_per_beat

            try:
                # Round onset to the nearest multiple of tatum
                note[0] = round(note[0]) * Fraction(1, mindiv)
                data[instrument_name].append(note)
                    
            except Exception as e:
                print(f"Error with onset calculation: {note[0]}, Track: {instrument_name}")
                raise e
    return data


def process_midi_separating_instruments(midi_file_path:str, beat_file_path:str, mindiv:int=24) -> dict:
    """Processes a MIDI file and returns note information (onset, pitch)
    separated by instruments with onsets normalized to beats.

    Args:
      midi_file_path: Path to the MIDI file.
      beat_file_path: Path to the beat file, beat as time float numbers, one per line.
      mindiv:int, 1/tatum (default is 24 to handle double semiquaver and triplet)

    Returns:
      A dictionaries with keys 'midi_file' with the file name, and one key per 'instrument' with a list of values [note onset, note pitch(, note duration)].
    """
    
    midi_data = pretty_midi.PrettyMIDI(midi_file_path)
    beat_times = _extract_beat_times(beat_file_path)

    data = dict()
    data['midi_file'] = os.path.basename(midi_file_path)

    for i, instrument in enumerate(midi_data.instruments):
        instrument_name = f"Track {i}"
        data[instrument_name] = list()

        for note in instrument.notes:
            onset_value = _find_nearest_division(note.start, beat_times, mindiv)

            data[instrument_name].append((onset_value, note.pitch))
                      
    return data



if __name__ == "__main__":
    midi_file_path = "/home/deck/Documents/Intership 2025/POP909-Dataset/POP909/001/001.mid"
    beat_file_path = "/home/deck/Documents/Intership 2025/POP909-Dataset/POP909/001/beat_midi.txt"
    data = process_midi_separating_instruments(midi_file_path, beat_file_path, 12)

    print(data.keys())