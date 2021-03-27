import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential as Base_Model
from tensorflow.keras.datasets import fashion_mnist as dataset
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


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
