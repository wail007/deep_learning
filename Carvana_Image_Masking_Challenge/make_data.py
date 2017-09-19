#!/usr/bin/env python
import os
import pdb
import argparse

import h5py
import numpy  as np

import matplotlib.pyplot as plt

from scipy import misc
from skimage import exposure

IMG_SIZE =  512
IMG_CNT  =  5088
X_DIR    = 'train'
Y_DIR    = 'train_masks'

def mean(a, batch_size=32):
    s = np.zeros([1,1,1,3], dtype=np.float64)

    for i in range(0, len(a), batch_size):
        s += np.sum(a[i:min(i + batch_size, len(a))], (0,1,2), dtype=np.float64)

    return s / (a.shape[0]*a.shape[1]*a.shape[2])


def minus(a, val, batch_size=32):
    for i in range(0, len(a), batch_size):
        a[i:min(i + batch_size, len(a))] -= val
        

def div(a, val, batch_size=32):
    for i in range(0, len(a), batch_size):
        a[i:min(i + batch_size, len(a))] /= val


def add_dset(file, dset_name, shape, img_list, interp):
    dset = file.create_dataset(dset_name, shape, dtype=np.uint8)
    for i, img in enumerate(img_list):
        dset[i] = misc.imread(img)[:,:,0:shape[-1]]
        print('%s: %d' % (dset_name, i), end='\r')
    
def add_seq_dset(file, dset_name, shape, seq_list, interp):
    dset = file.create_dataset(dset_name, shape, dtype=np.uint8)
    for i, seq in enumerate(seq_list):
        for j, frame in enumerate(seq):
            dset[i, j] = misc.imresize(misc.imread(frame), shape[-3:-1], interp)[:,:,0:shape[-1]]
        print('%s: %d' % (dset_name, i), end='\r')


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dim', type=int)
    parser.add_argument('-s', '--sequence', action='store_true')
    parser.add_argument('-o', '--output')
    args = parser.parse_args()

    x_list = np.sort(np.array([os.path.join(X_DIR, x) for x in os.listdir(X_DIR)])).reshape((-1, 16))
    y_list = np.sort(np.array([os.path.join(Y_DIR, y) for y in os.listdir(Y_DIR)])).reshape((-1, 16))

    shuffled_idx = np.random.permutation(len(x_list))

    x_list = x_list[shuffled_idx]
    y_list = y_list[shuffled_idx]

    if args.sequence:
        f = h5py.File(args.output, 'w')
        add_seq_dset(f, 'train/x', (x_list.shape[0], x_list.shape[1], args.dim, args.dim, 3), x_list, 'bilinear')
        add_seq_dset(f, 'train/y', (x_list.shape[0], x_list.shape[1], args.dim, args.dim, 1), y_list, 'nearest' )
        f.close()
    else:
        x_list = x_list.flatten()
        y_list = y_list.flatten()

        f = h5py.File(args.output, 'w')
        add_dset(f, 'train/x', (x_list.shape[0], 1280, 1918, 3), x_list, 'bilinear')
        add_dset(f, 'train/y', (x_list.shape[0], 1280, 1918, 1), y_list, 'nearest')
        f.close()

    """
    y_list = y_list.flatten()

    background_total = np.uint64(0)
    foreground_total = np.uint64(0)
    resolution       = np.uint64(1918*1280)
    for mask_path in y_list:
        mask = misc.imread(mask_path)[:,:,0]
        
        background = np.sum(mask == 0, dtype=np.uint64)

        background_total += background
        foreground_total += resolution - background

    
    print('Background: %d' % background_total)
    print('Foreground: %d' % foreground_total)
    """
    """
    f = h5py.File('train_seq_1024_eq.hdf5', 'r+')
    x_train_dset = f['train/x']
    
    for i, seq in enumerate(x_train_dset):
        for j, img in enumerate(seq):
            x_train_dset[i,j] = exposure.equalize_adapthist(img) * 255
        print(i, end='\r')
    
    f.close()
    """
    """
    f = h5py.File('train_%d.hdf5' % IMG_SIZE, 'r+')

    x_train_dset = f['train/x']
    x_mean_dset  = f.create_dataset('train/mean'  , (1, 1, 1, 3), dtype=np.float64)
    #x_stddev_dset = f.create_dataset('train/stddev', (1, 1, 1, 3), dtype=np.float64)
    x_mean_dset[:] = mean(x_train_dset)

    y_train_dset = f['train/y']
    weights = np.zeros((IMG_SIZE,IMG_SIZE), dtype=np.float32)
    for i, img in enumerate(y_train_dset):
        print(i, end='\r')

        diff_h = img[ :,1:,0] != img[:  ,:-1,0]
        diff_v = img[1:, :,0] != img[:-1,:  ,0]

        diff_h_before = diff_h.copy()
        diff_h_after  = diff_h.copy()
        diff_v_before = diff_v.copy()
        diff_v_after  = diff_v.copy()

        diff_h_before[img[ :  ,1:  ,0] == 0] = False
        diff_h_after [img[ :  , :-1,0] == 0] = False
        diff_v_before[img[1:  , :  ,0] == 0] = False
        diff_v_after [img[ :-1, :  ,0] == 0] = False

        edge = np.zeros(weights.shape, dtype=np.bool)
        edge[ :  , :-1][diff_h_after ] = True
        edge[1:  , :  ][diff_v_before] = True
        edge[ :-1, :  ][diff_v_after ] = True
        edge[ :  ,1:  ][diff_h_before] = True

        weights[edge] += 1

    weights = (weights - weights.min()) / weights.max()

    loss_weight_dset = f.create_dataset('train/loss_weight', (IMG_SIZE, IMG_SIZE), dtype=np.float32)
    loss_weight_dset[:] = weights

    f.close()
    """


if __name__ == '__main__':
    main()
