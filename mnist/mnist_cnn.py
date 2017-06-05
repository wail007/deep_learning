#!/usr/bin/env python
"""
Trains a CNN on the MNIST dataset.
Gets to 99.5% test accuracy
"""
import numpy  as np
import pandas as pd

from keras.datasets import mnist

from keras.models       import Sequential, load_model
from keras.layers       import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.layers.noise import GaussianNoise
from keras.optimizers   import Adam
from keras.callbacks    import ModelCheckpoint, EarlyStopping
from keras.utils        import to_categorical

def main():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.expand_dims(x_train, axis=3).astype('float32') / 255
    x_test  = np.expand_dims(x_test , axis=3).astype('float32') / 255
    y_train = to_categorical(y_train, 10)
    y_test  = to_categorical(y_test , 10)
    
    model = Sequential()
    model.add(GaussianNoise(stddev=0.3, input_shape=x_train.shape[1:]))
    model.add(Conv2D(32, 3, activation='relu'))
    model.add(Conv2D(32, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, 3, activation='relu'))
    model.add(Conv2D(64, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.summary()

    model.compile(optimizer=Adam(), 
                  loss="categorical_crossentropy", 
                  metrics=["accuracy"])

    chkpt = ModelCheckpoint(filepath='model.hdf5', 
                            monitor='val_acc', 
                            verbose=1, 
                            save_best_only=True, 
                            mode='max')

    early_stop = EarlyStopping(monitor='val_acc', 
                               patience=10, 
                               verbose=1, 
                               mode='max')

    model.fit(x_train, y_train,
              batch_size=128, 
              epochs=200,
              callbacks=[chkpt, early_stop], 
              validation_data=(x_test,y_test))
    
    model = load_model(filepath='model.hdf5')

    print(model.evaluate(x_test, y_test, batch_size=1000))


if __name__ == '__main__':
    main()

