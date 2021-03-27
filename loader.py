import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist as dataset
import os
import numpy as np
from sklearn.model_selection import train_test_split

def load_dataset():
    # Load Fashion MNIST datasets
    (x_train, y_train), (x_test, y_test) = dataset.load_data()

    # Brief information of datasets
    print("Shape of original training examples:", np.shape(x_train))
    print("Shape of original test examples:", np.shape(x_test))
    print("Shape of original training result:", np.shape(y_train))
    print("Shape of original test result:", np.shape(y_test))

    return (x_train, y_train), (x_test, y_test)


def load_dataset_expanddim():
    (x_train, y_train), (x_test, y_test) = dataset.load_data()

    # Expand dimensions of datasets
    x_train_1, x_test_1 = np.expand_dims(
        x_train, axis=3), np.expand_dims(x_test, axis=3)

    # Rescale the images from [0,255] to the [0.0, 1.0] range.
    x_train_1, x_test_1 = np.array(
        x_train, dtype=np.float32)/255.0, np.array(x_test, dtype=np.float32)/255.0
    y_train_1, y_test_1 = tf.keras.utils.to_categorical(
        y_train, 10, dtype=np.uint8), tf.keras.utils.to_categorical(y_test, 10, dtype=np.uint8)

    # Brief information of datasets
    print("Shape of original training examples:", np.shape(x_train_1))
    print("Shape of original test examples:", np.shape(x_test_1))
    print("Shape of original training result:", np.shape(y_train_1))
    print("Shape of original test result:", np.shape(y_test_1))

    return (x_train_1, y_train_1) , (x_test_1, y_test_1)
