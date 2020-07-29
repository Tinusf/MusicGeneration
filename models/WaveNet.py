import tensorflow.keras as keras
import tensorflow.keras.backend as K
from models.BaseModel import BaseModel


class WaveNet(BaseModel):
    def __init__(self, n):
        K.clear_session()
        model = keras.models.Sequential()

        # embedding layer
        model.add(keras.layers.Embedding(n, 100, input_length=32, trainable=True))

        model.add(keras.layers.Conv1D(64, 3, padding="causal", activation="relu"))
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.MaxPool1D(2))

        model.add(keras.layers.Conv1D(128, 3, activation="relu", dilation_rate=2, padding="causal"))
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.MaxPool1D(2))

        model.add(keras.layers.Conv1D(256, 3, activation="relu", dilation_rate=4, padding="causal"))
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.MaxPool1D(2))

        # model.add(keras.layers.Conv1D(256,5,activation="relu"))
        model.add(keras.layers.GlobalMaxPool1D())

        model.add(keras.layers.Dense(256, activation="relu"))
        model.add(keras.layers.Dense(n, activation="softmax"))

        model.compile(loss=keras.losses.sparse_categorical_crossentropy, optimizer="adam", metrics=["accuracy"])

        model.summary()
        self.model = model
