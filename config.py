# How many epochs should be trained on.
EPOCHS = 3
# Which model should be used. Options: "LSTM", "WaveNet"
WHICH_MODEL = "LSTM"
# Should you load the weights from the model.
SHOULD_LOAD_MODEL = True
# Where the models are stored.
MODEL_DIRECTORY = "saved_models/lstm/"
# Which model within the MODEL_DIRECTORY should be used for loading and saving the model.
MODEL_FILE = "model7.h5"
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
