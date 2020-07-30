import tensorflow.keras as keras
from models.BaseModel import BaseModel


class LSTM(BaseModel):
    def __init__(self, n_notes):
        model = keras.models.Sequential()
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.LSTM(256, return_sequences=True,
                                    kernel_regularizer=keras.regularizers.l2(1e-3)))
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.LSTM(256, kernel_regularizer=keras.regularizers.l2(1e-3)))
        # model.add(keras.layers.Dropout(0.2))
        # model.add(keras.layers.BatchNormalization())
        # model.add(keras.layers.Dense(256, kernel_regularizer=keras.regularizers.l2(1e-3)))
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation("relu"))
        model.add(keras.layers.Dense(n_notes, kernel_regularizer=keras.regularizers.l2(1e-3)))
        model.add(keras.layers.Activation("softmax"))
        optimizer = keras.optimizers.Adam(lr=0.001)
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=optimizer, metrics=["accuracy"])
        self.model = model
