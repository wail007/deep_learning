#!/usr/bin/env python
import os
import argparse
import h5py
import numpy as np

from scipy import misc

X_DIR = 'train'
Y_DIR = 'train_masks'


def add_dset(file, dset_name, shape, img_list):
    dset = file.create_dataset(dset_name, shape, dtype=np.uint8)
    for i, img in enumerate(img_list):
        dset[i] = misc.imread(img)[:, :, 0:shape[-1]]
        print('%s: %d' % (dset_name, i), end='\r')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dim', type=int)
    parser.add_argument('-o', '--output')
    args = parser.parse_args()

    x_list = np.sort(np.array([os.path.join(X_DIR, x) for x in os.listdir(X_DIR)])).reshape((-1, 16))
    y_list = np.sort(np.array([os.path.join(Y_DIR, y) for y in os.listdir(Y_DIR)])).reshape((-1, 16))

    shuffled_idx = np.random.permutation(len(x_list))

    x_list = x_list[shuffled_idx]
    y_list = y_list[shuffled_idx]

    x_list = x_list.flatten()
    y_list = y_list.flatten()

    f = h5py.File(args.output, 'w')
    add_dset(f, 'train/x', (x_list.shape[0], 1280, 1918, 3), x_list)
    add_dset(f, 'train/y', (x_list.shape[0], 1280, 1918, 1), y_list)
    f.close()


if __name__ == '__main__':
    main()
