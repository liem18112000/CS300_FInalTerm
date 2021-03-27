import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential as Base_Model
import numpy as np
from sklearn.model_selection import train_test_split


class Singleton:
    """
    A non-thread-safe helper class to ease implementing singletons.
    This should be used as a decorator -- not a metaclass -- to the
    class that should be a singleton.

    The decorated class can define one `__init__` function that
    takes only the `self` argument. Also, the decorated class cannot be
    inherited from. Other than that, there are no restrictions that apply
    to the decorated class.

    To get the singleton instance, use the `instance` method. Trying
    to use `__call__` will result in a `TypeError` being raised.

    """

    def __init__(self, decorated):
        self._decorated = decorated

    def instance(self):
        """
        Returns the singleton instance. Upon its first call, it creates a
        new instance of the decorated class and calls its `__init__` method.
        On all subsequent calls, the already created instance is returned.

        """
        try:
            return self._instance
        except AttributeError:
            self._instance = self._decorated()
            return self._instance

    def __call__(self):
        raise TypeError('Singletons must be accessed through `instance()`.')

    def __instancecheck__(self, inst):
        return isinstance(inst, self._decorated)


@Singleton
class ModelFactory(object):

    def createPlainModel(self):
        model = Base_Model([
            Flatten(input_shape=(28, 28)),
            Dense(512, activation='relu'),
            Dense(10, activation='softmax')
        ])

        model.summary()
        return model

    def createSimpleCNNModel(self):
        model = Base_Model([
            Input(shape=(28, 28, 1)),
            UpSampling3D(size=(1, 1, 3)),
            Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 3)),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(512, activation='relu'),
            Dense(10, activation='softmax')
        ])

        model.summary()
        return model

    def createCNNModel_1(self):
        model = Base_Model([
            Input(shape=(28, 28, 1)),
            UpSampling3D(size=(1, 1, 3)),
            Conv2D(48, (5, 5), activation='relu', padding="same", input_shape=(28, 28, 3)),
            MaxPooling2D((2, 2), padding="same"),
            Conv2D(96, (5, 5), activation='relu', padding="same"),
            MaxPooling2D((2, 2), padding="same"),
            Conv2D(80, (5, 5), activation='relu', padding="same"),
            Conv2D(96, (5, 5), activation='relu', padding="same"),
            Flatten(),
            Dense(512, activation='relu'),
            Dense(10, activation='softmax')
        ])

        model.summary()
        return model

    def createCNNModel_2(self):
        model = Base_Model([
            Input(shape=(28, 28, 1)),
            UpSampling3D(size=(1, 1, 3)),
            Conv2D(48, (5, 5), activation='relu', padding="same", input_shape=(28, 28, 3)),
            MaxPooling2D((2, 2), padding="same"),
            Conv2D(80, (5, 5), activation='relu', padding="same"),
            MaxPooling2D((2, 2), padding="same"),
            Conv2D(64, (5, 5), activation='relu', padding="same"),
            Conv2D(80, (5, 5), activation='relu', padding="same"),
            Conv2D(96, (5, 5), activation='relu', padding="same"),
            Conv2D(96, (5, 5), activation='relu', padding="same"),
            Conv2D(64, (5, 5), activation='relu', padding="same"),
            Conv2D(80, (5, 5), activation='relu', padding="same"),
            Flatten(),
            Dense(512, activation='relu'),
            Dense(10, activation='softmax')
        ])

        model.summary()
        return model

    def createMiniVGGModel_1(self):
        model = Base_Model([
            Input(shape=(28, 28, 1)),
            UpSampling3D(size=(1, 1, 3)),

            Conv2D(64, (3, 3), activation='relu',padding="same", input_shape=(28, 28, 3)),
            BatchNormalization(axis=1),
            Conv2D(64, (3, 3), activation='relu', padding="same"),
            BatchNormalization(axis=1),
            MaxPooling2D((2, 2)),
            Dropout(0.25),

            Conv2D(128, (3, 3), activation='relu', padding="same"),
            BatchNormalization(axis=1),
            Conv2D(128, (3, 3), activation='relu', padding="same"),
            BatchNormalization(axis=1),
            MaxPooling2D((2, 2)),
            Dropout(0.25),

            Conv2D(256, (3, 3), activation='relu', padding="same"),
            BatchNormalization(axis=1),
            Conv2D(256, (3, 3), activation='relu', padding="same"),         
            BatchNormalization(axis=1),
            MaxPooling2D((2, 2)),
            Dropout(0.25),

            Conv2D(512, (3, 3), activation='relu', padding="same"),
            BatchNormalization(axis=1),
            Conv2D(512, (3, 3), activation='relu', padding="same"),
            BatchNormalization(axis=1),
            MaxPooling2D((2, 2)),
            Dropout(0.25),

            Flatten(),
            Dense(512, activation='relu'),
            BatchNormalization(axis=1),

            Dense(10, activation='softmax')
        ])

        model.summary()
        return model


    def createMiniVGGModel_2(self):
        model = Base_Model([
            Input(shape=(28, 28, 1)),
            UpSampling3D(size=(1, 1, 3)),

            Conv2D(64, (3, 3), activation='relu',padding="same", input_shape=(28, 28, 3)),
            BatchNormalization(axis=1),
            Conv2D(64, (3, 3), activation='relu', padding="same"),
            BatchNormalization(axis=1),
            MaxPooling2D((2, 2)),
            Dropout(0.25),

            Conv2D(128, (3, 3), activation='relu', padding="same"),
            BatchNormalization(axis=1),
            Conv2D(128, (3, 3), activation='relu', padding="same"),
            BatchNormalization(axis=1),
            MaxPooling2D((2, 2)),
            Dropout(0.25),

            Conv2D(256, (3, 3), activation='relu', padding="same"),
            BatchNormalization(axis=1),
            Conv2D(256, (3, 3), activation='relu', padding="same"),         
            BatchNormalization(axis=1),
            Conv2D(256, (3, 3), activation='relu', padding="same"),
            BatchNormalization(axis=1),
            Conv2D(256, (3, 3), activation='relu', padding="same"),
            MaxPooling2D((2, 2)),
            Dropout(0.25),

            Conv2D(512, (3, 3), activation='relu', padding="same"),
            BatchNormalization(axis=1),
            Conv2D(512, (3, 3), activation='relu', padding="same"),
            BatchNormalization(axis=1),
            Conv2D(512, (3, 3), activation='relu', padding="same"),
            BatchNormalization(axis=1),
            Conv2D(512, (3, 3), activation='relu', padding="same"),
            BatchNormalization(axis=1),
            MaxPooling2D((2, 2)),
            Dropout(0.25),

            Flatten(),
            Dense(4096, activation='relu'),
            BatchNormalization(axis=1),
            Dense(4096, activation='relu'),
            BatchNormalization(axis=1),
            Dense(1024, activation='relu'),
            BatchNormalization(axis=1),

            Dense(10, activation='softmax')
        ])

        model.summary()
        return model
