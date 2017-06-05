#!/usr/bin/env python
import numpy  as np
import pandas as pd

from keras.datasets import mnist

from keras.models       import Sequential
from keras.layers       import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.layers.noise import GaussianNoise
from keras.optimizers   import Adam, Nadam
from keras.utils        import to_categorical

def main():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.expand_dims(x_train, axis=3).astype('float32') / 255
    x_test  = np.expand_dims(x_test , axis=3).astype('float32') / 255
    y_train = to_categorical(y_train, 10)
    y_test  = to_categorical(y_test , 10)

    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=3,
                     activation='relu', 
                     input_shape=x_train.shape[1:]))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10 , activation='softmax'))

    model.summary()

    model.compile(optimizer=Nadam(), 
                  loss="categorical_crossentropy", 
                  metrics=["accuracy"])

    model.fit(x_train, y_train,
              batch_size=128, epochs=30, validation_data=(x_test,y_test))

    print(model.evaluate(x_test, y_test))


if __name__ == '__main__':
    main()

