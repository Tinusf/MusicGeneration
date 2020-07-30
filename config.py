# How many epochs should be trained on.
EPOCHS = 50
# Which model should be used. Options: "LSTM", "WaveNet"
WHICH_MODEL = "LSTM"
# Should you load the weights from the model.
SHOULD_LOAD_MODEL = True
# Where the models are stored.
MODEL_DIRECTORY = "saved_models/lstm/"
# Which model within the MODEL_DIRECTORY should be used for loading and saving the model.
MODEL_FILE = "model_pianoviolin_onesong5.h5"
# Which directory the midi files are located in.
MIDI_DIRECTORY = "data/pianoviolin/"
# Should you just load the pickle files instead of parsing all the midi files.
LOAD_CACHED_MIDI_FILES = False
# Filter out infrequently used notes.
FILTER_INFREQUENTLY = False
# Use one hot encoding.
ONE_HOT = True
# When generating new music should you just use the greedy best choice, or should you use probabilities
# in order to get some random notes.
GREEDY_CHOICE = True
# Do not predict the same note twice in a row.
ENFORCE_NO_DUPLICATES = True
# How many notes in a row can be the same instrument. Set to -1 to disable.
SAME_INSTRUMENT_MAX_SEQUENCE_LIMIT = 16
# How many notes in a row needs to be the same instrument. seg to -1 to disable.
SAME_INSTRUMENT_MIN_SEQUENCE_LIMIT = 8
