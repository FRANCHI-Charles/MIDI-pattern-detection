import os
import re
from fractions import Fraction
import pretty_midi



def process_midi_separating_instruments(midi_file_path:str, mindiv:int=24, return_duration = False) -> dict:
    """Processes a MIDI file and returns note information (onset, pitch(, duration))
    separated by instruments with onsets normalized to beats.
    WARNING: Work only for midi with no tempos changes.

    Args:
      midi_file_path: Path to the MIDI file.
      mindiv:int, Minimum tatum (default is 24 to handle double semiquaver and triplet)
      return_duration: bool, If True, returns the duration of the notes in beats.

    Returns:
      A dictionaries with keys 'midi_file' with the file name, and one key per 'instrument' with a list of values [note onset, note pitch(, note duration)].
    """
    midi_data = pretty_midi.PrettyMIDI(midi_file_path)

    # Get the average BPM
    _, tempos = midi_data.get_tempo_changes()
    bpm = (tempos[0]) if len(tempos) > 0 else Fraction(120)

    # Conversion factor: seconds -> beats
    seconds_to_beats = bpm / 60

    # Define the tatum as 1 / (8 * 3)
    tatum = Fraction(1, mindiv)

    data = dict()
    data['midi_file'] = os.path.basename(midi_file_path)

    for instrument in midi_data.instruments:
        instrument_name = instrument.name if instrument.name else 'Not provided'
        data[instrument_name] = list()

        for note in instrument.notes:
            onset_in_beats = note.start * seconds_to_beats
            duration_in_beats = (note.end - note.start) * seconds_to_beats

            try:
                # Round onset to the nearest multiple of tatum
                onset_value = round(onset_in_beats / tatum) * tatum
                duration_value = round(duration_in_beats / tatum) * tatum

                if return_duration:
                    data[instrument_name].append([
                        onset_value,
                        note.pitch,
                        duration_value
                    ])
                else:
                    # Append only the onset and pitch
                    data[instrument_name].append([
                        onset_value,
                        note.pitch
                    ])
                    
            except Exception as e:
                print(f"Error with onset calculation: {onset_in_beats}, Track: {instrument_name}, Note: {note.pitch}")
                raise e

    return data


def extract_midi_info(midi_path):
    "Extract the instruments of a midi file."
    midi = pretty_midi.PrettyMIDI(midi_path)

    info = {'File Name': os.path.basename(midi_path)}

    for instrument in midi.instruments:
        track_name = instrument.name if instrument.name else 'Unknown'
        #instrument_name = pretty_midi.program_to_instrument_name(instrument.program) if not instrument.is_drum else 'Drum Kit'

        info[track_name] = True

    return info
