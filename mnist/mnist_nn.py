#!/usr/bin/env python
"""
Trains a deep NN on the MNIST dataset.
Gets to 98.80% test accuracy
"""
import numpy  as np
import pandas as pd

from keras.datasets import mnist

from keras.models       import Sequential, load_model
from keras.layers       import Dense
from keras.layers.noise import GaussianNoise
from keras.optimizers   import Adam 
from keras.callbacks    import ModelCheckpoint, EarlyStopping
from keras.utils        import to_categorical

def main():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784).astype('float32') / 255.0
    x_test  = x_test .reshape(10000, 784).astype('float32') / 255.0
    y_train = to_categorical(y_train, 10)
    y_test  = to_categorical(y_test , 10)

    model = Sequential()
    model.add(GaussianNoise(stddev=0.4, input_shape=(x_train.shape[1],)))
    model.add(Dense(512, activation="relu"))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(10 , activation="softmax"))

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
                               patience=20, 
                               verbose=1, 
                               mode='max')
    
    model.fit(x_train, 
              y_train,
              batch_size=128, 
              epochs=200, 
              callbacks=[chkpt, early_stop], 
              validation_data=(x_test,y_test))
    

    model = load_model(filepath='model.hdf5')

    print(model.metrics_names, model.evaluate(x_test, y_test, batch_size=x_test.shape[0]))

if __name__ == '__main__':
    main()