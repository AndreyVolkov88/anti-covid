import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import models


def create_model(img_shape):

    inputs = layers.Input(shape=img_shape)

    conv1 = layers.Conv3D(32, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.Conv3D(32, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPool3D(pool_size=(2, 2, 2))(conv1)
    conv2 = layers.Conv3D(64, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv3D(64, 3, activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPool3D(pool_size=(2, 2, 2))(conv2)

    conv3 = layers.Conv3D(128, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.Conv3D(128, 3, activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPool3D(pool_size=(2, 2, 2))(conv3)

    conv4 = layers.Conv3D(256, 3, activation='relu', padding='same')(pool3)
    conv4 = layers.Conv3D(256, 3, activation='relu', padding='same')(conv4)
    pool4 = layers.MaxPool3D(pool_size=(2, 2, 2))(conv4)
    flat = layers.Flatten()(pool4)
    dense1 = layers.Dense(256*((img_shape[0]/2**4)**2)*(img_shape[2]/2**4), activation='relu')(flat)
    # dense2 = layers.Dense(256*((img_shape[0]/2**4)**2)*(img_shape[2]/2**4), activation='relu')(dense1)
    out = layers.Dense(2, activation='sigmoid')(dense1)
    model = keras.Model(inputs=inputs, outputs=out)
    model.compile(optimizer=optimizers.Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return model
