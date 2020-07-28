import tensorflow.keras.callbacks as callbacks
import tensorflow.keras as keras
import os
import config
from abc import ABC


class BaseModel(ABC):
    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    def train_if_necessary(self, x_train, y_train, x_val=None, y_val=None):
        model_path = config.MODEL_DIRECTORY + config.MODEL_FILE
        if config.SHOULD_LOAD_MODEL and (os.path.exists(model_path)):
            self.model = keras.models.load_model(model_path)
        else:
            validation_data = None
            callback_list = []
            checkpoint_loss = callbacks.ModelCheckpoint(f"best_loss.h5",
                                                        monitor='loss', save_best_only=True,
                                                        mode='min')
            callback_list.append(checkpoint_loss)
            if x_val:
                checkpoint_val_acc = callbacks.ModelCheckpoint(f"best_val_acc.h5",
                                                               monitor='val_accuracy', save_best_only=True,
                                                               mode='max')

                checkpoint_val_loss = callbacks.ModelCheckpoint(f"best_val_loss.h5",
                                                                monitor='val_loss', save_best_only=True,
                                                                mode='min')
                callback_list.append(checkpoint_val_acc)
                callback_list.append(checkpoint_val_loss)
                validation_data = (x_val, y_val)
            self.model.fit(x_train, y_train, epochs=config.EPOCHS,
                           validation_data=validation_data, callbacks=callback_list)
            self.model.save(model_path)
