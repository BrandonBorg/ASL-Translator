import numpy as np
import tensorflow as tf
from tensorflow import keras


def train_model(x_train, y_train):
    model = keras.Sequential()

    # Layer 1
    model.add(keras.layers.Conv2D(128, kernel_size=(5, 5), input_shape=(48, 48, 1), padding='valid'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))

    # Layer 2
    model.add(keras.layers.Conv2D(256, kernel_size=(3, 3), padding='valid'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))

    # Layer 3
    model.add(keras.layers.Conv2D(512, kernel_size=(3, 3), padding='valid'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))

    # Layer 4
    model.add(keras.layers.Conv2D(1024, kernel_size=(1, 1), padding='valid'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))

    model.add(keras.layers.Flatten())

    # Layer 5
    model.add(keras.layers.Dense(1024))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dropout(0.5))

    # Layer 6
    model.add(keras.layers.Dense(1024))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dropout(0.5))

    # Layer 7
    model.add(keras.layers.Dense(29))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('softmax'))

    model.compile(optimizer ='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    hist = model.fit(x_train, y_train, epochs=50)

    print(hist.history)
    model.save('CNN_POOLING_TESTING2.h5')
