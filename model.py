import os
import tensorflow.keras as keras


class lstm():
    def __init__(self, n_notes):
        self.model_path = "model.h5"

        model = keras.models.Sequential()
        model.add(keras.layers.LSTM(128, return_sequences=True))
        model.add(keras.layers.LSTM(128))
        model.add(keras.layers.Dense(256))
        model.add(keras.layers.Activation("relu"))
        model.add(keras.layers.Dense(n_notes))
        model.add(keras.layers.Activation("softmax"))
        model.compile(loss=keras.losses.categorical_crossentropy, optimizer="adam")
        self.model = model

    def train(self, x, y, epochs):
        if (os.path.exists(self.model_path)):
            self.model = keras.models.load_model(self.model_path)
        else:
            self.model.fit(x, y, epochs=epochs)
            self.model.save(self.model_path)
