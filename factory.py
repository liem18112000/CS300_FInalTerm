from tensorflow.python.keras import backend as K
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential as Base_Model
import numpy as np
from sklearn.model_selection import train_test_split
from layers.dropblock import DropBlock2D


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

    def createMiniResNetModel(self, num_filters=64, num_blocks=4, num_sub_blocks=2,use_max_pool=False):

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
                x = DropBlock2D(keep_prob=0.8, block_size=1)(x)

            num_filters = 2 * num_filters

        # Add classifier on top.
        x = AveragePooling2D()(x)
        y = Flatten()(x)
        outputs = Dense(10,activation='softmax', kernel_initializer='he_normal')(y)

        # Instantiate model.
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        model.summary()
        return model


def _bernoulli(shape, mean):
    return tf.nn.relu(tf.sign(mean - tf.random_uniform(shape, minval=0, maxval=1, dtype=tf.float32)))


class DropBlock2D(tf.keras.layers.Layer):
    def __init__(self, keep_prob, block_size, scale=True, **kwargs):
        super(DropBlock2D, self).__init__(**kwargs)
        self.keep_prob = float(keep_prob) if isinstance(
            keep_prob, int) else keep_prob
        self.block_size = int(block_size)
        self.scale = tf.constant(scale, dtype=tf.bool) if isinstance(
            scale, bool) else scale

    def compute_output_shape(self, input_shape):
        return input_shape

    def build(self, input_shape):
        assert len(input_shape) == 4
        _, self.h, self.w, self.channel = input_shape.as_list()
        # pad the mask
        p1 = (self.block_size - 1) // 2
        p0 = (self.block_size - 1) - p1
        self.padding = [[0, 0], [p0, p1], [p0, p1], [0, 0]]
        self.set_keep_prob()
        super(DropBlock2D, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        def drop():
            mask = self._create_mask(tf.shape(inputs))
            output = inputs * mask
            output = tf.cond(self.scale,
                             true_fn=lambda: output *
                             tf.to_float(tf.size(mask)) / tf.reduce_sum(mask),
                             false_fn=lambda: output)
            return output

        if training is None:
            training = K.learning_phase()
        output = tf.cond(tf.logical_or(tf.logical_not(training), tf.equal(self.keep_prob, 1.0)),
                         true_fn=lambda: inputs,
                         false_fn=drop)
        return output

    def set_keep_prob(self, keep_prob=None):
        """This method only supports Eager Execution"""
        if keep_prob is not None:
            self.keep_prob = keep_prob
        w, h = tf.to_float(self.w), tf.to_float(self.h)
        self.gamma = (1. - self.keep_prob) * (w * h) / (self.block_size ** 2) / \
                     ((w - self.block_size + 1) * (h - self.block_size + 1))

    def _create_mask(self, input_shape):
        sampling_mask_shape = tf.stack([input_shape[0],
                                        self.h - self.block_size + 1,
                                        self.w - self.block_size + 1,
                                        self.channel])
        mask = _bernoulli(sampling_mask_shape, self.gamma)
        mask = tf.pad(mask, self.padding)
        mask = tf.nn.max_pool(mask, [1, self.block_size, self.block_size, 1], [
                              1, 1, 1, 1], 'SAME')
        mask = 1 - mask
        return mask


class DropBlock3D(tf.keras.layers.Layer):
    def __init__(self, keep_prob, block_size, scale=True, **kwargs):
        super(DropBlock3D, self).__init__(**kwargs)
        self.keep_prob = float(keep_prob) if isinstance(
            keep_prob, int) else keep_prob
        self.block_size = int(block_size)
        self.scale = tf.constant(scale, dtype=tf.bool) if isinstance(
            scale, bool) else scale

    def compute_output_shape(self, input_shape):
        return input_shape

    def build(self, input_shape):
        assert len(input_shape) == 5
        _, self.d, self.h, self.w, self.channel = input_shape.as_list()
        # pad the mask
        p1 = (self.block_size - 1) // 2
        p0 = (self.block_size - 1) - p1
        self.padding = [[0, 0], [p0, p1], [p0, p1], [p0, p1], [0, 0]]
        self.set_keep_prob()
        super(DropBlock3D, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        def drop():
            mask = self._create_mask(tf.shape(inputs))
            output = inputs * mask
            output = tf.cond(self.scale,
                             true_fn=lambda: output *
                             tf.to_float(tf.size(mask)) / tf.reduce_sum(mask),
                             false_fn=lambda: output)
            return output

        if training is None:
            training = K.learning_phase()
        output = tf.cond(tf.logical_or(tf.logical_not(training), tf.equal(self.keep_prob, 1.0)),
                         true_fn=lambda: inputs,
                         false_fn=drop)
        return output

    def set_keep_prob(self, keep_prob=None):
        """This method only supports Eager Execution"""
        if keep_prob is not None:
            self.keep_prob = keep_prob
        d, w, h = tf.to_float(self.d), tf.to_float(self.w), tf.to_float(self.h)
        self.gamma = ((1. - self.keep_prob) * (d * w * h) / (self.block_size ** 3) /
                      ((d - self.block_size + 1) * (w - self.block_size + 1) * (h - self.block_size + 1)))

    def _create_mask(self, input_shape):
        sampling_mask_shape = tf.stack([input_shape[0],
                                        self.d - self.block_size + 1,
                                        self.h - self.block_size + 1,
                                        self.w - self.block_size + 1,
                                        self.channel])
        mask = _bernoulli(sampling_mask_shape, self.gamma)
        mask = tf.pad(mask, self.padding)
        mask = tf.nn.max_pool3d(mask, [
                                1, self.block_size, self.block_size, self.block_size, 1], [1, 1, 1, 1, 1], 'SAME')
        mask = 1 - mask
        return mask

