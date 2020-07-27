from typing import Tuple
from collections import Counter
import numpy as np
from music21 import *
import os
import pickle


def load_all_midi_files() -> np.ndarray:
    pickle_path = "midi_files_array.pickle"
    if (os.path.exists(pickle_path)):
        return pickle.load(open(pickle_path, "br"))

    # Data downloaded from https://www.analyticsvidhya.com/blog/2020/01/how-to-perform-automatic-music-generation/
    path = "data/"
    files = [i for i in os.listdir(path) if i.endswith(".mid")]
    array = np.array([read_midi(path + i) for i in files])
    pickle.dump(array, open(pickle_path, "bw"))
    return array


# defining function to read MIDI files
def read_midi(file: str) -> np.ndarray:
    print("Loading Music File:", file)

    notes = []
    notes_to_parse = None

    # parsing a midi file
    midi = converter.parse(file)

    # grouping based on different instruments
    s2 = instrument.partitionByInstrument(midi)

    # Looping over all the instruments
    for part in s2.parts:

        # select elements of only piano
        if 'Piano' in str(part):

            notes_to_parse = part.recurse()

            # finding whether a particular element is note or a chord
            for element in notes_to_parse:

                # note
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))

                # chord
                elif isinstance(element, chord.Chord):
                    # 2.5.9
                    notes.append('.'.join(str(n) for n in element.normalOrder))

    return np.array(notes)


def get_frequent_notes(notes: np.ndarray) -> list:
    # Flatten the notes array.
    notes_1d = [element for note_ in notes for element in note_]
    # Frequency dictionary.
    freq = dict(Counter(notes_1d))
    # Return the notes that have a count of 50 or above.
    return [note_ for note_, count in freq.items() if count >= 50]


def filter_frequent_notes(notes: np.ndarray, frequent_notes: list) -> np.ndarray:
    return np.array([[_note for _note in one_song if _note in frequent_notes] for one_song in notes])


def create_training_dataset(notes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    This method creates a dataset containing 32 notes as input and a single note as output for those 32 notes.
    :param notes: Numpy array of notes.
    :return: X, y dataset.
    """
    no_of_timesteps = 32
    x = []
    y = []

    for _note in notes:
        for i in range(0, len(_note) - no_of_timesteps, 1):
            input_notes = _note[i:i + no_of_timesteps]
            output_note = _note[i + no_of_timesteps]

            x.append(input_notes)
            y.append(output_note)

    return np.array(x), np.array(y)
