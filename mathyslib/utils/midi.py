import os
import re
from fractions import Fraction
import pretty_midi



def process_midi_separating_instruments(midi_file_path:str, mindiv:int=24) -> dict:
    """Processes a MIDI file and returns note information (onset, pitch, duration)
    separated by instruments with onsets normalized to beats.

    Args:
      midi_file_path: Path to the MIDI file.
      mindiv:int, Minimum tatum (default is 24 to handle double semiquaver and triplet)

    Returns:
      A dictionaries with keys 'midi_file' with the file name, and one key per 'instrument' with a list of values [note onset, note pitch, note duration].
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
                rounded_onset = round(onset_in_beats / tatum) * tatum
                rounded_duration = round(duration_in_beats / tatum) * tatum

                # Convert to float if denominator is not a multiple of 3, 7 or 11
                _divisable = lambda frac : frac.denominator % 3 != 0 and frac.denominator % 7 != 0 and frac.denominator % 11 != 0
                onset_value = float(rounded_onset) if _divisable(rounded_onset) else rounded_onset
                duration_value = float(rounded_duration) if _divisable(rounded_duration) else rounded_duration

                data[instrument_name].append([
                    onset_value,
                    note.pitch,
                    duration_value
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
