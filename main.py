import os
from tensorflow.keras.utils import to_categorical
from data_util import DataUtil
import numpy as np
import music_util
from sklearn.model_selection import train_test_split
from model import lstm
import tensorflow.keras.backend as K


def main():
    arr = music_util.load_all_midi_files()
    frequent_notes = music_util.get_frequent_notes(arr)
    n_notes = len(frequent_notes) + 1
    arr_filtered = music_util.filter_frequent_notes(arr, frequent_notes)

    x, y = music_util.create_training_dataset(arr_filtered)
    # Because this dataset is a mix of string like "A4" and sequenes like "2-5-9" we need to transform them
    # to indices in order to get numbers.
    dataUtil = DataUtil()
    x, y = dataUtil.transform_data_index(x, y)

    x = x.reshape((x.shape[0], x.shape[1], 1))
    y = y.reshape((y.shape[0]))
    x = to_categorical(x, num_classes=n_notes)
    y = to_categorical(y, num_classes=n_notes)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = lstm(n_notes)
    model.train(x_train, y_train, epochs=3)

    random_melody = generate_random(model.model, x_test, n_notes)
    random_melody = [dataUtil.index_to_elem[x] for x in random_melody]
    print(random_melody)
    music_util.convert_to_midi(random_melody)


def generate_random(model, x_test, n_classes, no_of_timesteps=32):
    ind = np.random.randint(0, len(x_test) - 1)
    random_music = x_test[ind]
    predictions = []
    for i in range(100):
        # random_music = random_music.reshape(1, no_of_timesteps)
        prob = model.predict(np.expand_dims(random_music, axis=0)).squeeze()
        y_pred = np.random.choice(np.arange(len(prob)), p=prob)
        # y_pred = np.argmax(prob, axis=0)
        y_one_hot = to_categorical(y_pred, num_classes=n_classes)
        predictions.append(y_pred)
        random_music = np.insert(random_music, 32, y_one_hot, axis=0)
        random_music = random_music[1:]

    return predictions


if __name__ == '__main__':
    main()
