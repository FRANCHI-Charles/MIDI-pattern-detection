from fractions import Fraction

def get_track_notes_nbr(track:dict) -> dict:
    """
    Get the number of notes for each instrument.

    Parameters
    ----------
    track : dict
        A dict of instrument name as key and a list of notes as value.

    Returns
    -------
    dict
        A dict of instrument name as key and the number of notes as value.
    """
    output = dict()
    for instrument, notes in track.items():
        if not instrument == "midi_file":
            output[instrument] = len(notes)
    return output

def get_track_length(track:dict) -> dict:
    """
    Get the length of each instrument.

    Parameters
    ----------
    track : dict
        A dict of instrument name as key and a list of notes as value.

    Returns
    -------
    dict
        A dict of instrument name as key and the length of the track as value.
    """
    output = dict()
    for instrument, notes in track.items():
        if not instrument == "midi_file":
            onsets = [note[0] for note in notes]
            output[instrument] = max(onsets) - min(onsets)
        
    return output


def get_track_extreme_pitch(track:dict) -> dict:
    """
    Get the extreme pitch for each instrument.

    Parameters
    ----------
    track : dict
        A dict of instrument name as key and a list of notes as value.

    Returns
    -------
    dict
        A dict of instrument name as key and the extreme pitch as value.
    """
    output = dict()
    for instrument, notes in track.items():
        if not instrument == "midi_file":
            output[instrument] = (min([note[1] for note in notes]), max([note[1] for note in notes]))
    return output


def non_binary_notes(track:dict) -> dict:
    """
    Poportion of non binary rythm notes.
    Only works if the notes are Fractions.

    Parameters
    ----------
    track : dict
        A dict of instrument name as key and a list of notes as value.

    Returns
    -------
    float
        The proportion of non binary rythm notes.
    """
    output = 0
    total = 0
    for instrument, notes in track.items():
        if not instrument == "midi_file":
            for note in notes:
                if not isinstance(note[0], Fraction):
                    print("Warning: Some notes are not fractions. The function will return False.")
                    return None
                
                if note[0].denominator % 3 == 0 or note[0].denominator % 7 == 0 or note[0].denominator % 11 == 0:
                    output += 1
            total += len(notes)
                
    return output/total