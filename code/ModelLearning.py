import numpy as np
import tensorflow as tf
from tensorflow import keras


def train_model(x_train, y_train):
    model = keras.Sequential([
        keras.layers.Conv2D(64, kernel_size=3, activation='relu', input_shape=(48, 48, 1)),
        keras.layers.Conv2D(32, kernel_size=3, activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(29, activation='softmax')
    ])

    model.compile(optimizer ='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    hist = model.fit(x_train, y_train, epochs=4) # seems like its capping at 5 at 98%

    print(hist.history)
    model.save('64_32_Conv_4epoch.h5')
