#!/usr/bin/env python
import pdb
import argparse

import numpy                as np
import matplotlib.pyplot    as plt

from scipy                     import misc
from keras.utils.io_utils      import HDF5Matrix
from keras.preprocessing.image import DirectoryIterator
import keras.backend as K

import models
from iterators import HDF5MatrixIterator
 
def dice_coef_on_batch(y_true, y_pred, axis=(2,3,4)):
    threshold = 0.5
    smooth    = 1e-5

    y_true = y_true > threshold
    y_pred = y_pred > threshold

    return np.mean((2. * np.sum(y_true * y_pred, axis=axis) + smooth) / (np.sum(y_true, axis=axis) + np.sum(y_pred, axis=axis) + smooth))

def predict_seq_on_batch(model, seq, window_size):
    pred = np.zeros(seq.shape[:-1])
    pred = np.expand_dims(pred, axis=len(seq.shape)-1)

    for i in range(seq.shape[1]):
        window = np.arange(i, i + window_size) % seq.shape[1]
        pred[:,window] += model.predict_on_batch(seq[:,window]) > 0.5 

    return pred / window_size

def eval_seq_on_batch(model, seq_img, seq_mask, window_size):
    return dice_coef_on_batch(predict_seq_on_batch(model, seq_img, window_size), seq_mask)

def evaluate_seq_generator(model, gen, window_size):
    seq_cnt = 0.
    score   = 0.
    i = 1
    for batch_x, batch_y in gen:
        score   += eval_seq_on_batch(model, batch_x, batch_y, window_size) * len(batch_x)
        seq_cnt += len(batch_x)
        print("Batch: %d, Score: %f" % (i,score/seq_cnt), end='\r')
        i += 1

    return score / seq_cnt
        
 
def preprocess_seq(batch_x, batch_y):
    return (batch_x.astype(K.floatx()) / 255, batch_y.astype(K.floatx()) / 255)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', required=True)
    parser.add_argument('-i', '--input', required=True)
    args = parser.parse_args()

    model = models.get_seq_unet_1024()
    if args.model:
        model.load_weights(args.model)

    x  = HDF5Matrix(args.input, "train/x", start=255)
    y  = HDF5Matrix(args.input, "train/y", start=255)
    it = HDF5MatrixIterator(x, y, batch_size=4, preprocess=preprocess_seq)

    score = evaluate_seq_generator(model, it, 5)
    
    print(score)


if __name__ == '__main__':
    main()
