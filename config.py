# How many epochs should be trained on.
EPOCHS = 350
# Which model should be used. Options: "LSTM", "WaveNet"
WHICH_MODEL = "WaveNet"
# Should you load the weights from the model.
SHOULD_LOAD_MODEL = True
# Where the models are stored.
MODEL_DIRECTORY = "saved_models/wavenet/"
# Which model within the MODEL_DIRECTORY should be used for loading and saving the model.
MODEL_FILE = "model.h5"
# Filter out infrequently used notes.
FILTER_INFREQUENTLY = False
# Use one hot encoding.
ONE_HOT = False
# When generating new music should you just use the greedy best choice, or should you use probabilities
# in order to get some random notes.
GREEDY_CHOICE = True
