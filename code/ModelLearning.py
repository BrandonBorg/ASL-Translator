import numpy as np
import tensorflow as tf
from tensorflow import keras


def train_model(x_train, y_train):
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(29, activation='softmax')
    ])

    model.compile(optimizer ='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=10)

    model.save('test_model.h5')
