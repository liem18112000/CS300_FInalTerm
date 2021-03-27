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
            UpSampling3D(size=(1, 1, 1)),
            Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(512, activation='relu'),
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
            Dense(512, activation='relu'),
            Dense(10, activation='softmax')
        ])

        model.summary()
        return model


    def createMiniVGGModel(self):
        model = Base_Model([
            Input(shape=(28, 28, 1)),
            UpSampling3D(size=(1, 1, 1)),

            Conv2D(64, (3, 3), activation='relu',padding="same", input_shape=(28, 28, 1)),
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

    def createMiniResNetModel(self):
        input = Input(shape=(28, 28, 1))

        '''block_1'''
        b1_cnv2d_1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), padding='same',
                            use_bias=False, name='b1_cnv2d_1', kernel_initializer='normal')(input)
        b1_relu_1 = ReLU(name='b1_relu_1')(b1_cnv2d_1)
        b1_bn_1 = BatchNormalization(
            epsilon=1e-3, momentum=0.999, name='b1_bn_1')(b1_relu_1)  # size: 14*14

        b1_cnv2d_2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(2, 2), padding='same',
                            use_bias=False, name='b1_cnv2d_2', kernel_initializer='normal')(b1_bn_1)
        b1_relu_2 = ReLU(name='b1_relu_2')(b1_cnv2d_2)
        b1_out = BatchNormalization(
            epsilon=1e-3, momentum=0.999, name='b1_out')(b1_relu_2)  # size: 14*14

        '''block 2'''
        b2_cnv2d_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same',
                            use_bias=False, name='b2_cnv2d_1', kernel_initializer='normal')(b1_out)
        b2_relu_1 = ReLU(name='b2_relu_1')(b2_cnv2d_1)
        b2_bn_1 = BatchNormalization(
            epsilon=1e-3, momentum=0.999, name='b2_bn_1')(b2_relu_1)  # size: 14*14

        b2_add = add([b1_out, b2_bn_1])  #

        b2_cnv2d_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same',
                            use_bias=False, name='b2_cnv2d_2', kernel_initializer='normal')(b2_add)
        b2_relu_2 = ReLU(name='b2_relu_2')(b2_cnv2d_2)
        b2_out = BatchNormalization(
            epsilon=1e-3, momentum=0.999, name='b2_bn_2')(b2_relu_2)  # size: 7*7

        '''block 3'''
        b3_cnv2d_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same',
                            use_bias=False, name='b3_cnv2d_1', kernel_initializer='normal')(b2_out)
        b3_relu_1 = ReLU(name='b3_relu_1')(b3_cnv2d_1)
        b3_bn_1 = BatchNormalization(
            epsilon=1e-3, momentum=0.999, name='b3_bn_1')(b3_relu_1)  # size: 7*7

        b3_add = add([b2_out, b3_bn_1])  #

        b3_cnv2d_2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same',
                            use_bias=False, name='b3_cnv2d_2', kernel_initializer='normal')(b3_add)
        b3_relu_2 = ReLU(name='b3_relu_2')(b3_cnv2d_2)
        b3_out = BatchNormalization(
            epsilon=1e-3, momentum=0.999, name='b3_out')(b3_relu_2)  # size: 3*3

        '''block 4'''
        b4_cnv2d_1 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same',
                            use_bias=False, name='b4_cnv2d_1', kernel_initializer='normal')(b3_out)
        b4_relu_1 = ReLU(name='b4_relu_1')(b4_cnv2d_1)
        b4_bn_1 = BatchNormalization(
            epsilon=1e-3, momentum=0.999, name='b4_bn_1')(b4_relu_1)  # size: 3*3

        b4_add = add([b3_out, b4_bn_1])  #

        b4_cnv2d_2 = Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), padding='same',
                            use_bias=False, name='b4_cnv2d_2', kernel_initializer='normal')(b4_add)
        b4_relu_2 = ReLU(name='b4_relu_2')(b4_cnv2d_2)
        b4_out = BatchNormalization(
            epsilon=1e-3, momentum=0.999, name='b4_out')(b4_relu_2)  # size: 1*1

        '''block 5'''
        b5_avg_p = GlobalAveragePooling2D()(b4_out)
        output = Dense(10, name='model_output', activation='softmax',
                    kernel_initializer='he_uniform')(b5_avg_p)

        model = tf.keras.Model(input, output)

        model.summary()
        return model
