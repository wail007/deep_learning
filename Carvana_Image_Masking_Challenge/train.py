#!/usr/bin/env python
import pdb
import math
import random
import argparse
import numpy as np

from functools import partial

from keras.callbacks            import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image  import apply_transform
from keras.utils.io_utils       import HDF5Matrix
from keras                      import backend as K

from skimage.color    import rgb2gray
from skimage.exposure import equalize_hist, equalize_adapthist, rescale_intensity


import models

from iterators import HDF5MatrixIterator

import matplotlib.pyplot as plt

BATCH_SIZE = 1
SEQ_SIZE   = 3
CROP_SIZE  = 1280
DATA_FILE  = 'train_1280x1918.hdf5'


def transform(img, rot_angle, trans_v, trans_h):
    transform = np.array([[math.cos(rot_angle), -math.sin(rot_angle), trans_v],
                          [math.sin(rot_angle),  math.cos(rot_angle), trans_h],
                          [                 0.,                   0.,      1.]])
    return apply_transform(img, transform, 2, 'constant', 0.)


def preprocess_train(batch_x, batch_y):
    batch_x = batch_x.astype(K.floatx()) / 255
    batch_y = batch_y.astype(K.floatx()) / 255

    batch_x = np.array([equalize_adapthist(x) for x in batch_x])
    
    crop_v  = np.random.randint(0, batch_x.shape[1] - CROP_SIZE + 1, BATCH_SIZE)
    crop_h  = np.random.randint(0, batch_x.shape[2] - CROP_SIZE + 1, BATCH_SIZE)
    batch_x = np.array([img[v:v+CROP_SIZE,h:h+CROP_SIZE] for img, v, h in zip(batch_x, crop_v, crop_h)])
    batch_y = np.array([img[v:v+CROP_SIZE,h:h+CROP_SIZE] for img, v, h in zip(batch_y, crop_v, crop_h)])
    
    choice = np.random.choice(2, BATCH_SIZE, replace=True)
    batch_x[choice == 1] = np.flip(batch_x[choice == 1], axis=2)
    batch_y[choice == 1] = np.flip(batch_y[choice == 1], axis=2)
    """ 
    choice = np.random.choice(3, BATCH_SIZE, replace=True)
    if 1 in choice:
        batch_x[choice == 1] = np.array([equalize_hist(x)                     for x in batch_x[choice == 1]])
    if 2 in choice:
        batch_x[choice == 2] = np.array([equalize_adapthist(x)                for x in batch_x[choice == 2]])
    if 3 in choice:
        batch_x[choice == 3] = np.array([np.stack([rgb2gray(x)] * 3, axis=-1) for x in batch_x[choice == 3]])
    if 4 in choice:
        batch_x[choice == 4] = np.array([rescale_intensity(x)                 for x in batch_x[choice == 4]])
    """
    """
    pad     = ((0,0),(0,0),(1,1),(0,0))
    batch_x = np.pad(batch_x, pad, 'constant')
    batch_y = np.pad(batch_y, pad, 'constant')
    """
    
    """
    r = np.random.choice(2, BATCH_SIZE, replace=True)
    v = np.random.uniform(-0.1 * batch_x.shape[2], 0.1 * batch_x.shape[2], BATCH_SIZE)
    h = np.random.uniform(-0.1 * batch_x.shape[2], 0.1 * batch_x.shape[2], BATCH_SIZE)
    if 1 in r:
        batch_x[r == 1] = np.array([transform(x, 0., trans_v, trans_h) for x, trans_v, trans_h in zip(batch_x[r == 1], v[r == 1], h[r == 1])])
        batch_y[r == 1] = np.array([transform(y, 0., trans_v, trans_h) for y, trans_v, trans_h in zip(batch_y[r == 1], v[r == 1], h[r == 1])])
    """
    return (batch_x, batch_y)


def preprocess_val(batch_x, batch_y):
    batch_x = batch_x.astype(K.floatx()) / 255
    batch_y = batch_y.astype(K.floatx()) / 255

    batch_x = np.array([equalize_adapthist(x) for x in batch_x])
    
    crop_v  = np.random.randint(0, batch_x.shape[1] - CROP_SIZE + 1, BATCH_SIZE)
    crop_h  = np.random.randint(0, batch_x.shape[2] - CROP_SIZE + 1, BATCH_SIZE)
    batch_x = np.array([img[v:v+CROP_SIZE,h:h+CROP_SIZE] for img, v, h in zip(batch_x, crop_v, crop_h)])
    batch_y = np.array([img[v:v+CROP_SIZE,h:h+CROP_SIZE] for img, v, h in zip(batch_y, crop_v, crop_h)])
    
    """
    pad = ((0, 0), (0, 0), (1, 1), (0, 0))
    batch_x = np.pad(batch_x, pad, 'constant')
    batch_y = np.pad(batch_y, pad, 'constant')
    """
    return (batch_x, batch_y)


def preprocess_train_seq(batch_x, batch_y):
    shift = random.randint(0, batch_x.shape[1] - 1)
    batch_x = np.roll(batch_x, shift, axis=1)
    batch_y = np.roll(batch_y, shift, axis=1)

    seq_start = random.randint(0, batch_x.shape[1] - SEQ_SIZE)
    batch_x = batch_x[:, seq_start:seq_start + SEQ_SIZE]
    batch_y = batch_y[:, seq_start:seq_start + SEQ_SIZE]

    if random.random() < 0.5:
        batch_x = np.flip(batch_x, axis=1)
        batch_y = np.flip(batch_y, axis=1)

    if random.random() < 0.5:
        batch_x = np.flip(batch_x, axis=3)
        batch_y = np.flip(batch_y, axis=3)

    batch_x = batch_x.astype(K.floatx()) / 255
    batch_y = batch_y.astype(K.floatx()) / 255

    rot_angle = 0.
    translate_v = 0.
    translate_h = 0.
    #if random.random() < 0.5:
    #    rot_angle = math.radians((random.random() * 2. - 1.) * 5.0)
    if random.random() < 0.5:
        translate_v = (random.random() * 0.1 - 0.05) * batch_x.shape[1]
        translate_h = (random.random() * 0.1 - 0.05) * batch_x.shape[2]
    if rot_angle or translate_v or translate_h:
        transform = np.array([[math.cos(rot_angle), -math.sin(rot_angle), translate_v],
                              [math.sin(rot_angle), math.cos(rot_angle), translate_h],
                              [0., 0., 1.]])
        shape_x = batch_x.shape
        shape_y = batch_y.shape
        batch_x = np.array([apply_transform(x, transform, 2, 'constant', 0.) for seq in batch_x for x in seq]).reshape(shape_x)
        batch_y = np.array([apply_transform(y, transform, 2, 'constant', 0.) for seq in batch_y for y in seq]).reshape(shape_y)

    return (batch_x, batch_y)


def preprocess_val_seq(batch_x, batch_y):
    seq_start = random.randint(0, batch_x.shape[1] - SEQ_SIZE)
    batch_x = batch_x[:, seq_start:seq_start + SEQ_SIZE]
    batch_y = batch_y[:, seq_start:seq_start + SEQ_SIZE]

    batch_x = batch_x.astype(K.floatx()) / 255
    batch_y = batch_y.astype(K.floatx()) / 255

    return (batch_x, batch_y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model')
    args = parser.parse_args()

    model = models.get_unet_1024(input_shape=(1280, 1280, 3))
    if args.model:
        model.load_weights(args.model, by_name=True)

    """
    train_mean            = HDF5Matrix(DATA_FILE, 'train/mean')[:]
    preprocess_train_func = partial(preprocess_train, train_mean)
    preprocess_val_func   = partial(preprocess_val  , train_mean)
    """

    train_x   = HDF5Matrix(DATA_FILE, "train/x", end=4080)
    train_y   = HDF5Matrix(DATA_FILE, "train/y", end=4080)
    train_it  = HDF5MatrixIterator(train_x, train_y, batch_size=BATCH_SIZE, preprocess=preprocess_train, shuffle=True)
    train_cnt = len(train_x)

    val_x   = HDF5Matrix(DATA_FILE, "train/x", start=4080)
    val_y   = HDF5Matrix(DATA_FILE, "train/y", start=4080)
    val_it  = HDF5MatrixIterator(val_x, val_y, batch_size=BATCH_SIZE, preprocess=preprocess_val)
    val_cnt = len(val_x)

    chkpt = ModelCheckpoint(filepath='model.hdf5',
                            monitor='val_dice_coef',
                            verbose=1,
                            save_best_only=True,
                            save_weights_only=True,
                            mode='max')

    early_stop = EarlyStopping(monitor='val_dice_coef',
                               patience=5,
                               verbose=1,
                               mode='max')

    model.fit_generator(generator=train_it,
                        steps_per_epoch=math.ceil(train_cnt / BATCH_SIZE),
                        epochs=200,
                        callbacks=[chkpt, early_stop],
                        validation_data=val_it,
                        validation_steps=math.ceil(val_cnt / BATCH_SIZE),
                        max_q_size=20,
                        workers=4)


if __name__ == '__main__':
    main()
