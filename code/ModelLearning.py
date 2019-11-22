import numpy as np
import tensorflow as tf
from tensorflow import keras


def train_model(x_train, y_train):
    model = keras.Sequential([
        keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu', input_shape=(48, 48, 1), padding='valid'),
        keras.layers.MaxPooling2D(pool_size=(2,2), strides=2, padding='valid'),

        keras.layers.Conv2D(64, kernel_size=3, activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'),

        keras.layers.Conv2D(64, kernel_size=3, activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'),

        keras.layers.Flatten(),
        keras.layers.Dense(29, activation='softmax')
    ])

    model.compile(optimizer ='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    hist = model.fit(x_train, y_train, epochs=4) # seems like its capping at 5 at 98%

    print(hist.history)
    model.save('3_pool_CNN_4epoch.h5')
