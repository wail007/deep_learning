#!/usr/bin/env python
import os, pdb

import h5py
import numpy  as np
import pandas as pd

from keras.preprocessing.image import load_img, img_to_array

from scipy import misc

IMAGE_SIZE = 128
TRAIN_DIR_CAT = "train/cat"
TRAIN_DIR_DOG = "train/dog"

def main():
    img_list = np.array([os.path.join(TRAIN_DIR_CAT, img) for img in os.listdir(TRAIN_DIR_CAT)] + 
                        [os.path.join(TRAIN_DIR_DOG, img) for img in os.listdir(TRAIN_DIR_DOG)])
    img_cnt  = len(img_list)
    img_list = img_list[np.random.permutation(img_cnt)]

    f = h5py.File("cats_vs_dogs_" + str(IMAGE_SIZE) + ".hdf5", 'w')
    train_y = f.create_dataset("train/y", (img_cnt,), dtype=np.uint8)
    train_x = f.create_dataset("train/x", (img_cnt,IMAGE_SIZE,IMAGE_SIZE,3), dtype=np.uint8)

    
    for i, img in enumerate(img_list):
        train_x[i] = misc.imresize(misc.imread(img), (IMAGE_SIZE,IMAGE_SIZE,3))
        train_y[i] = 0 if "cat" in img else 1

    f.close()

if __name__ == '__main__':
    main()