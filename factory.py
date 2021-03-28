from tensorflow.python.keras import backend as K
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import l2
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
            Dense(filter * 8, activation='relu'),
            Dense(10, activation='softmax')
        ])

        model.summary()
        return model

    def createSimpleCNNModel(self):
        model = Base_Model([
            Input(shape=(28, 28, 1)),
            UpSampling3D(size=(1, 1, 1)),
            Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(filter * 8, activation='relu'),
            Dense(10, activation='softmax')
        ])

        model.summary()
        return model

    def createCNNModel(self):
        model = Base_Model([
            Input(shape=(28, 28, 1)),
            UpSampling3D(size=(1, 1, 1)),
            Conv2D(48, (5, 5), activation='relu', padding="same", input_shape=(28, 28, 1)),
            MaxPooling2D((2, 2), padding="same"),
            Conv2D(96, (5, 5), activation='relu', padding="same"),
            MaxPooling2D((2, 2), padding="same"),
            Conv2D(80, (5, 5), activation='relu', padding="same"),
            Conv2D(96, (5, 5), activation='relu', padding="same"),
            Flatten(),
            Dense(filter * 8, activation='relu'),
            Dense(10, activation='softmax')
        ])

        model.summary()
        return model

    def createPretrainModel(self, pretrain_model, trainable = False, reszie_rate=2):

        core = pretrain_model(
            include_top=False, weights='imagenet', input_shape=(28 * reszie_rate, 28 * reszie_rate, 3)
        )

        for layer in core.layers:
            layer.trainable = trainable
        
        model = Base_Model([
            Input(shape=(28, 28, 1)),
            UpSampling3D(size=(reszie_rate, reszie_rate, 3)),
            core,
            Dropout(0.5),
            Flatten(),
            Dense(4096, activation='relu'),
            Dense(4096, activation='relu'),
            Dense(10, activation='softmax'),
        ])

        model.summary()
        return model


    def createMiniVGGModel(self, filter = 128):
        model = Base_Model([
            Input(shape=(28, 28, 1)),
            Conv2D(filter, (3, 3), activation='relu',padding="same", input_shape=(28, 28, 1)),
            BatchNormalization(axis=1),
            Conv2D(filter, (3, 3), activation='relu', padding="same"),
            BatchNormalization(axis=1),
            MaxPooling2D((2, 2)),
            Dropout(0.5),

            Conv2D(filter * 2, (3, 3), activation='relu', padding="same"),
            BatchNormalization(axis=1),
            Conv2D(filter * 2, (3, 3), activation='relu', padding="same"),
            BatchNormalization(axis=1),
            MaxPooling2D((2, 2)),
            Dropout(0.5),

            Conv2D(filter * 4, (3, 3), activation='relu', padding="same"),
            BatchNormalization(axis=1),
            Conv2D(filter * 4, (3, 3), activation='relu', padding="same"),         
            BatchNormalization(axis=1),
            Conv2D(filter * 4, (3, 3), activation='relu', padding="same"),
            BatchNormalization(axis=1),
            Conv2D(filter * 4, (3, 3), activation='relu', padding="same"),
            BatchNormalization(axis=1),
            MaxPooling2D((2, 2)),
            Dropout(0.5),

            Conv2D(filter * 8, (3, 3), activation='relu', padding="same"),
            BatchNormalization(axis=1),
            Conv2D(filter * 8, (3, 3), activation='relu', padding="same"),
            BatchNormalization(axis=1),
            Conv2D(filter * 8, (3, 3), activation='relu', padding="same"),
            BatchNormalization(axis=1),
            Conv2D(filter * 8, (3, 3), activation='relu', padding="same"),
            BatchNormalization(axis=1),
            MaxPooling2D((2, 2)),
            Dropout(0.5),

            Flatten(),
            Dense(filter * 8, activation='relu'),
            BatchNormalization(axis=1),
            Dense(filter * 8, activation='relu'),
            BatchNormalization(axis=1),

            Dense(10, activation='softmax')
        ])

        model.summary()
        return model

    def createMiniResNetModel(self, num_filters=64, num_blocks=4, num_sub_blocks=2,use_max_pool=False, use_dropout = False):

        # Creating model based on ResNet published archietecture
        inputs = Input(shape=(28, 28, 1))
        x = Conv2D(num_filters, padding='same',
                kernel_initializer='he_normal',
                kernel_size=7, strides=2,
                kernel_regularizer=l2(1e-4))(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        #Check by applying max pooling later (setting it false as size of image is small i.e. 28x28)
        if use_max_pool:
            x = MaxPooling2D(pool_size=3, padding='same', strides=2)(x)
            num_blocks = 3

        # Creating Conv base stack
        # Instantiate convolutional base (stack of blocks).
        for i in range(num_blocks):
            for j in range(num_sub_blocks):
                strides = 1
                is_first_layer_but_not_first_block = j == 0 and i > 0
                if is_first_layer_but_not_first_block:
                    strides = 2
                #Creating residual mapping using y
                y = Conv2D(num_filters,
                        kernel_size=3,
                        padding='same',
                        strides=strides,
                        kernel_initializer='he_normal',
                        kernel_regularizer=l2(1e-4))(x)
                y = BatchNormalization()(y)
                y = Activation('relu')(y)
                y = Conv2D(num_filters,
                        kernel_size=3,
                        padding='same',
                        kernel_initializer='he_normal',
                        kernel_regularizer=l2(1e-4))(y)
                y = BatchNormalization()(y)
                if is_first_layer_but_not_first_block:
                    x = Conv2D(num_filters,
                            kernel_size=1,
                            padding='same',
                            strides=2,
                            kernel_initializer='he_normal',
                            kernel_regularizer=l2(1e-4))(x)
                #Adding back residual mapping
                x = add([x, y])
                x = Activation('relu')(x)

                # Add Dropout
                if use_dropout:
                    x = Dropout(0.25)(x)

            num_filters = 2 * num_filters

        # Add classifier on top.
        x = AveragePooling2D()(x)
        y = Flatten()(x)
        outputs = Dense(10,activation='softmax', kernel_initializer='he_normal')(y)

        # Instantiate model.
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        model.summary()
        return model


