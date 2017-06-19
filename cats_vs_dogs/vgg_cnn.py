#!/usr/bin/env python
import pdb
import math
import random
import h5py
import numpy as np

from keras                      import backend as K
from keras.models               import Sequential
from keras.layers               import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.optimizers           import SGD, Adam
from keras.initializers         import RandomNormal
from keras.preprocessing.image  import ImageDataGenerator, Iterator
from keras.utils.io_utils       import HDF5Matrix

import matplotlib.pyplot as plt

BATCH_SIZE  = 128
DATA_DIM    = 64
INPUT_SHAPE = (48,48,3)

class ImageIterator(Iterator):
    def __init__(self, x, y, shape=None, random_patch=False, 
                 flip_prob=0.5, batch_size=32, shuffle=False, seed=None):
        self.x = x
        self.y = y
        self.shape = x.shape[1:] if shape is None else shape
        self.random_patch = random_patch
        self.flip_prob = flip_prob
        super(ImageIterator, self).__init__(x.shape[0], batch_size, shuffle, seed)

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)

        #pdb.set_trace()

        batch_x = np.zeros([current_batch_size] + list(self.shape), dtype=K.floatx())
        batch_y = self.y[index_array]

        for i, j in enumerate(index_array):
            x = self.x[j]

            if self.random_patch:
                y_pos = random.randint(0, x.shape[0] - self.shape[0])
                x_pos = random.randint(0, x.shape[1] - self.shape[1])
            else: #centered
                y_pos = (x.shape[0] - self.shape[0]) // 2
                x_pos = (x.shape[1] - self.shape[1]) // 2

            x = x[y_pos:y_pos+self.shape[0],x_pos:x_pos+self.shape[1]]

            if random.random() < self.flip_prob:
                x = np.fliplr(x)
            
            batch_x[i] = x
        
        return batch_x, batch_y

def main():
    normal = RandomNormal(mean=0.0, stddev=0.01)

    model = Sequential()
    """
    model.add(Conv2D(filters=48, kernel_size=7, strides=2, padding='same', activation='relu', kernel_initializer=normal, input_shape=INPUT_SHAPE))
    model.add(MaxPooling2D(pool_size=3, strides=2))
    model.add(Conv2D(filters=128, kernel_size=5, padding='same', activation='relu', kernel_initializer=normal, bias_initializer='ones'))
    model.add(MaxPooling2D(pool_size=3, strides=2))
    model.add(Conv2D(filters=192, kernel_size=3, padding='same', activation='relu', kernel_initializer=normal))
    model.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu', kernel_initializer=normal, bias_initializer='ones'))
    model.add(MaxPooling2D(pool_size=3, strides=2))
    model.add(Flatten())
    model.add(Dense(2048, activation='relu', kernel_initializer=normal, bias_initializer='ones'))
    model.add(Dropout(0.5))
    model.add(Dense(2048, activation='relu', kernel_initializer=normal, bias_initializer='ones'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid', kernel_initializer=normal, bias_initializer='ones'))
    """
    model.add(Dense(1, activation='sigmoid', input_shape=(DATA_DIM*DATA_DIM*3,)))

    model.summary()

    model.compile(optimizer=SGD(lr=0.01), 
                  loss="binary_crossentropy",
                  metrics=["accuracy"])


    f = h5py.File("cats_vs_dogs_" + str(DATA_DIM) + ".hdf5", 'r')
    train_cnt = math.ceil(f["train/x"].shape[0] * 0.7)
    train_x   = f["train/x"][:train_cnt].astype(K.floatx()) / 255
    train_y   = f["train/y"][:train_cnt].astype(K.floatx())
    val_x     = f["train/x"][train_cnt:].astype(K.floatx()) / 255
    val_y     = f["train/y"][train_cnt:].astype(K.floatx())
    f.close()

    train_it = ImageIterator(train_x, train_y, shape=INPUT_SHAPE,
                             random_patch=True, batch_size=BATCH_SIZE, shuffle=True)
    val_it   = ImageIterator(val_x  , val_y  , shape=INPUT_SHAPE,
                             flip_prob=0.0, batch_size=BATCH_SIZE)
    
    model.fit_generator(generator=train_it,
                        steps_per_epoch=math.ceil(len(train_x)/BATCH_SIZE),
                        epochs=200,
                        validation_data=val_it,
                        validation_steps=math.ceil(len(val_x)/BATCH_SIZE),
                        workers=1)

if __name__ == '__main__':
    main()
