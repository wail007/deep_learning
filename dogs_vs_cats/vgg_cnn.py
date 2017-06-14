#!/usr/bin/env python
import os, pdb, math

import numpy as np
from  scipy import ndimage

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

BATCH_SIZE = 128

def main():
    """
    model = Sequential()
    model.add(Conv2D(filters=64 , kernel_size=3, padding='same', activation='relu', input_shape=(256,256,3)))
    model.add(MaxPool2D(pool_size=2))
    model.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=2))
    model.add(Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=2))
    model.add(Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=2))
    model.add(Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(units=4096, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(units=4096, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(units=1, activation='sigmoid'))
    """

    model = Sequential()
    model.add(Conv2D(32, 3, activation='relu', input_shape=(128,128,3)))
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
    model.add(Dense(1, activation='sigmoid'))

    model.summary()

    model.compile(optimizer=SGD(), 
                  loss="binary_crossentropy",
                  metrics=["accuracy"])


    train_gen = ImageDataGenerator().flow_from_directory('train',
                                                         target_size=(128, 128),
                                                         class_mode='binary',
                                                         batch_size=BATCH_SIZE)
    val_gen   = ImageDataGenerator().flow_from_directory('val',
                                                         target_size=(128, 128),
                                                         class_mode='binary',
                                                         batch_size=BATCH_SIZE)

    print(train_gen.class_indices)
    
    model.fit_generator(generator=train_gen,
                        steps_per_epoch=math.ceil(train_gen.samples/BATCH_SIZE),
                        epochs=200,
                        validation_data=val_gen,
                        validation_steps=math.ceil(val_gen.samples/BATCH_SIZE))

if __name__ == '__main__':
    main()