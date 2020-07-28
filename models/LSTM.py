import tensorflow.keras as keras
from models.BaseModel import BaseModel


class LSTM(BaseModel):
    def __init__(self, n_notes):
        model = keras.models.Sequential()
        model.add(keras.layers.LSTM(128, return_sequences=True))
        model.add(keras.layers.LSTM(128))
        model.add(keras.layers.Dense(256))
        model.add(keras.layers.Activation("relu"))
        model.add(keras.layers.Dense(n_notes))
        model.add(keras.layers.Activation("softmax"))
        optimizer = keras.optimizers.Adam(lr=0.01)
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=optimizer, metrics=["accuracy"])
        self.model = model
