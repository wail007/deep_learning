#!/usr/bin/env python
import math
import argparse
import numpy as np

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import apply_transform
from keras.utils.io_utils import HDF5Matrix
from keras import backend as K

import models

from iterators import HDF5MatrixIterator

BATCH_SIZE = 1
SEQ_SIZE = 3
CROP_SIZE = 1280
DATA_FILE = 'train_1280x1918.hdf5'


def transform(img, rot_angle, trans_v, trans_h):
    transform = np.array([[math.cos(rot_angle), -math.sin(rot_angle), trans_v],
                          [math.sin(rot_angle), math.cos(rot_angle), trans_h],
                          [0., 0., 1.]])
    return apply_transform(img, transform, 2, 'constant', 0.)


def zoom(img, zx, zy, shift_x, shift_y):
    transform = np.array([[zx, 0, shift_x],
                          [0, zy, shift_y],
                          [0, 0, 1]])
    return apply_transform(img, transform, 2, 'constant', 0.)


def preprocess_train(batch_x, batch_y):
    batch_x = batch_x.astype(K.floatx()) / 255
    batch_y = batch_y.astype(K.floatx()) / 255

    crop_v = np.random.randint(0, batch_x.shape[1] - CROP_SIZE + 1, BATCH_SIZE)
    crop_h = np.random.randint(0, batch_x.shape[2] - CROP_SIZE + 1, BATCH_SIZE)
    batch_x = np.array([img[v:v + CROP_SIZE, h:h + CROP_SIZE] for img, v, h in zip(batch_x, crop_v, crop_h)])
    batch_y = np.array([img[v:v + CROP_SIZE, h:h + CROP_SIZE] for img, v, h in zip(batch_y, crop_v, crop_h)])

    c = np.random.choice(2, BATCH_SIZE, replace=True)
    if 1 in c:
        batch_x[c == 1] = np.flip(batch_x[c == 1], axis=2)
        batch_y[c == 1] = np.flip(batch_y[c == 1], axis=2)

    c = np.random.choice(2, BATCH_SIZE, replace=True)
    if 1 in c:
        zx_list = np.random.uniform(0.75, 1.0, BATCH_SIZE)
        zy_list = np.random.uniform(0.75, 1.0, BATCH_SIZE)
        sx_list = np.array([np.random.randint(0, int((1.0 - z) * batch_x.shape[1]) + 1) for z in zx_list])
        sy_list = np.array([np.random.randint(0, int((1.0 - z) * batch_x.shape[1]) + 1) for z in zy_list])
        batch_x[c == 1] = np.array([zoom(x, zx, zy, sx, sy) for x, zx, zy, sx, sy in
                                    zip(batch_x[c == 1], zx_list[c == 1], zy_list[c == 1], sx_list[c == 1],
                                        sy_list[c == 1])])
        batch_y[c == 1] = np.array([zoom(y, zx, zy, sx, sy) for y, zx, zy, sx, sy in
                                    zip(batch_y[c == 1], zx_list[c == 1], zy_list[c == 1], sx_list[c == 1],
                                        sy_list[c == 1])])

    r = np.random.choice(2, BATCH_SIZE, replace=True)
    v = np.random.uniform(-0.1 * batch_x.shape[2], 0.1 * batch_x.shape[2], BATCH_SIZE)
    h = np.random.uniform(-0.1 * batch_x.shape[2], 0.1 * batch_x.shape[2], BATCH_SIZE)
    if 1 in r:
        batch_x[r == 1] = np.array(
            [transform(x, 0., trans_v, trans_h) for x, trans_v, trans_h in zip(batch_x[r == 1], v[r == 1], h[r == 1])])
        batch_y[r == 1] = np.array(
            [transform(y, 0., trans_v, trans_h) for y, trans_v, trans_h in zip(batch_y[r == 1], v[r == 1], h[r == 1])])

    return (batch_x, batch_y)


def preprocess_val(batch_x, batch_y):
    batch_x = batch_x.astype(K.floatx()) / 255
    batch_y = batch_y.astype(K.floatx()) / 255

    crop_v = np.random.randint(0, batch_x.shape[1] - CROP_SIZE + 1, BATCH_SIZE)
    crop_h = np.random.randint(0, batch_x.shape[2] - CROP_SIZE + 1, BATCH_SIZE)
    batch_x = np.array([img[v:v + CROP_SIZE, h:h + CROP_SIZE] for img, v, h in zip(batch_x, crop_v, crop_h)])
    batch_y = np.array([img[v:v + CROP_SIZE, h:h + CROP_SIZE] for img, v, h in zip(batch_y, crop_v, crop_h)])

    return (batch_x, batch_y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model')
    args = parser.parse_args()

    model = models.get_unet(input_shape=(1280, 1280, 3), pool_cnt=7, filter_cnt=8)
    if args.model:
        model.load_weights(args.model, by_name=True)

    train_x = HDF5Matrix(DATA_FILE, "train/x", end=4080)
    train_y = HDF5Matrix(DATA_FILE, "train/y", end=4080)
    train_it = HDF5MatrixIterator(train_x, train_y, batch_size=BATCH_SIZE, preprocess=preprocess_train, shuffle=True)
    train_cnt = len(train_x)

    val_x = HDF5Matrix(DATA_FILE, "train/x", start=4080)
    val_y = HDF5Matrix(DATA_FILE, "train/y", start=4080)
    val_it = HDF5MatrixIterator(val_x, val_y, batch_size=BATCH_SIZE, preprocess=preprocess_val)
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
