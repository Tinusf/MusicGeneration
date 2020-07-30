import config
from tensorflow.keras.utils import to_categorical
from data_util import DataUtil
import numpy as np
import music_util
from sklearn.model_selection import train_test_split
from models.LSTM import LSTM
from models.WaveNet import WaveNet


def main():
    notes_array = music_util.load_all_midi_files()

    if config.FILTER_INFREQUENTLY:
        frequent_notes = music_util.get_frequent_notes(notes_array)
        n_notes = len(frequent_notes)
        notes_array = music_util.filter_frequent_notes(notes_array, frequent_notes)
    else:
        freq = music_util.get_frequency_dict(notes_array)
        n_notes = len(freq)
    print(n_notes)
    x, y = music_util.create_training_dataset(notes_array)
    # Because this dataset is a mix of string like "A4" and sequenes like "2-5-9" we need to transform them
    # to indices in order to get numbers.
    dataUtil = DataUtil()
    x, y = dataUtil.transform_data_index(x, y)
    if config.ONE_HOT:
        x, y = dataUtil.one_hot(x, y, n_notes)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    if config.WHICH_MODEL == "WaveNet":
        model = WaveNet(n_notes)
    elif config.WHICH_MODEL == "LSTM":
        model = LSTM(n_notes)
    else:
        raise ValueError(config.WHICH_MODEL + " is not supported.")

    model.train_if_necessary(x_train, y_train, x_test, y_test)

    random_melody = generate_random(model.model, x_test, n_notes, dataUtil)
    random_melody = [dataUtil.index_to_elem[x] for x in random_melody]
    print(random_melody)
    music_util.convert_to_midi(random_melody)


def generate_random(model, x_test, n_classes, dataUtil):
    ind = np.random.randint(0, len(x_test) - 1)
    random_music = x_test[ind]
    predictions = []
    cur_instrument = ""
    cur_instrument_count = 0
    prev_note_i = None
    for i in range(100):
        prob = model.predict(np.expand_dims(random_music, axis=0)).squeeze()
        if config.GREEDY_CHOICE:
            if config.ENFORCE_NO_DUPLICATES and prev_note_i is not None:
                prob[prev_note_i] = 0

            y_pred = np.argmax(prob, axis=0)

            if config.SAME_INSTRUMENT_MIN_SEQUENCE_LIMIT > 0:
                if cur_instrument_count < config.SAME_INSTRUMENT_MIN_SEQUENCE_LIMIT:
                    note = dataUtil.index_to_elem[y_pred]
                    instrument = "violin" if note.startswith("violin") else "piano"
                    if instrument != cur_instrument:
                        # Filter out other instruments.
                        for index, note in dataUtil.index_to_elem.items():
                            if not note.startswith(cur_instrument):
                                prob[index] = 0
                        # Redo the prediction with other instruments filtered out.
                        y_pred = np.argmax(prob, axis=0)

            if config.SAME_INSTRUMENT_MAX_SEQUENCE_LIMIT > 0:
                note = dataUtil.index_to_elem[y_pred]
                instrument = "violin" if note.startswith("violin") else "piano"
                if instrument == cur_instrument:
                    cur_instrument_count += 1
                else:
                    cur_instrument_count = 0
                    cur_instrument = instrument
                if cur_instrument_count >= config.SAME_INSTRUMENT_MAX_SEQUENCE_LIMIT:
                    cur_instrument_count = 0
                    # set probabilities of that instrument to 0.
                    for index, note in dataUtil.index_to_elem.items():
                        if note.startswith(cur_instrument):
                            prob[index] = 0
                    # Redo the prediction with one instrument filtered out.
                    y_pred = np.argmax(prob, axis=0)
                    note = dataUtil.index_to_elem[y_pred]
                    cur_instrument = "violin" if note.startswith("violin") else "piano"
            prev_note_i = y_pred

        else:
            y_pred = np.random.choice(np.arange(len(prob)), p=prob)
        if config.ONE_HOT:
            y_one_hot = to_categorical(y_pred, num_classes=n_classes)
            random_music = np.insert(random_music, 32, y_one_hot, axis=0)
        else:
            random_music = np.insert(random_music, 32, y_pred, axis=0)
        random_music = random_music[1:]
        predictions.append(y_pred)

    return predictions


if __name__ == '__main__':
    main()
