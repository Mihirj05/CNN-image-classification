import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

CIFAR10_LABELS = [
    'airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'
]

def load_cifar10(normalize=True, val_split=0.1):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    # create a validation split from training set
    num_val = int(len(x_train) * val_split)
    x_val = x_train[:num_val]
    y_val = y_train[:num_val]
    x_train = x_train[num_val:]
    y_train = y_train[num_val:]

    if normalize:
        x_train = x_train.astype('float32') / 255.0
        x_val = x_val.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def create_generators(x_train, y_train, x_val=None, y_val=None, batch_size=32, augment=False):
    if augment:
        train_gen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1
        ).flow(x_train, y_train, batch_size=batch_size)
    else:
        train_gen = ImageDataGenerator().flow(x_train, y_train, batch_size=batch_size)

    val_gen = None
    if x_val is not None and y_val is not None:
        val_gen = ImageDataGenerator().flow(x_val, y_val, batch_size=batch_size)

    return train_gen, val_gen

def prep_custom_directory(data_dir, target_size=(32,32), batch_size=32, class_mode='sparse', shuffle=True):
    datagen = ImageDataGenerator(rescale=1./255)
    train_gen = datagen.flow_from_directory(
        os.path.join(data_dir, 'train'),
        target_size=target_size,
        batch_size=batch_size,
        class_mode=class_mode,
        shuffle=shuffle
    )
    val_gen = datagen.flow_from_directory(
        os.path.join(data_dir, 'val'),
        target_size=target_size,
        batch_size=batch_size,
        class_mode=class_mode,
        shuffle=False
    )
    return train_gen, val_gen
