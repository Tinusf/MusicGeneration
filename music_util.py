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
                    notes.append('.'.join(str(n) for n in element.normalOrder))

    return np.array(notes)
