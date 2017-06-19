#!/usr/bin/env python
import math, random
import h5py
import numpy as np

from keras.models               import Sequential
from keras.layers               import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.optimizers           import Adam
from keras.callbacks            import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image  import Iterator
from keras.utils.io_utils       import HDF5Matrix
from keras                      import backend as K

class HDF5MatrixIterator(Iterator):
    def __init__(self, x, y, batch_size=32, preprocess=None, shuffle=False, seed=None):
        self.x = x
        self.y = y
        self.shuffle = shuffle
        self.preprocess = preprocess
        super(HDF5MatrixIterator, self).__init__(x.shape[0], batch_size, shuffle, seed)

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)

        if self.shuffle:
            batch_x = np.array([self.x[int(i)] for i in index_array])
            batch_y = np.array([self.y[int(i)] for i in index_array])
        else:
            batch_x = self.x[index_array[0]:index_array[-1]]
            batch_y = self.y[index_array[0]:index_array[-1]]

        if self.preprocess is not None:
            batch_x = self.preprocess(batch_x)
        
        return batch_x, batch_y


def preprocess_func(batch_x):
    for i in range(len(batch_x)):
        if random.random() < 0.5:
            batch_x[i] = np.fliplr(batch_x[i])
    return batch_x


def main():
    BATCH_SIZE  = 128
    INPUT_SHAPE = (64,64,3)
    DATA_PATH   = "cats_vs_dogs_64.hdf5"

    model = Sequential()
    model.add(Conv2D(64, 3, activation='relu', input_shape=INPUT_SHAPE))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(128, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(256, 3, activation='relu'))
    model.add(Conv2D(256, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.summary()

    model.compile(optimizer=Adam(), 
                  loss="binary_crossentropy",
                  metrics=["accuracy"])

    train_y   = HDF5Matrix(DATA_PATH, "train/y", end=20000)
    train_x   = HDF5Matrix(DATA_PATH, "train/x", end=20000, 
                           normalizer=lambda x: x.astype(K.floatx()) / 255)
    train_it  = HDF5MatrixIterator(train_x, train_y, preprocess=preprocess_func,
                                   batch_size=BATCH_SIZE, shuffle=False)
    train_cnt = len(train_x)

    val_y   = HDF5Matrix(DATA_PATH, "train/y", start=20000)
    val_x   = HDF5Matrix(DATA_PATH, "train/x", start=20000, 
                         normalizer=lambda x: x.astype(K.floatx()) / 255)
    val_it  = HDF5MatrixIterator(val_x, val_y, batch_size=BATCH_SIZE, shuffle=False)
    val_cnt = len(val_x)

    chkpt = ModelCheckpoint(filepath='model.hdf5', 
                            monitor='val_acc', 
                            verbose=1, 
                            save_best_only=True, 
                            mode='max')

    early_stop = EarlyStopping(monitor='val_acc', 
                               patience=10, 
                               verbose=1, 
                               mode='max')
    
    model.fit_generator(generator=train_it,
                        steps_per_epoch=math.ceil(train_cnt/BATCH_SIZE),
                        epochs=200,
                        callbacks=[chkpt,early_stop],
                        validation_data=val_it,
                        validation_steps=math.ceil(val_cnt/BATCH_SIZE),
                        max_q_size=20,
                        workers=8)

if __name__ == '__main__':
    main()