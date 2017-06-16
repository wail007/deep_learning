import os, pdb

import h5py
import numpy  as np
import pandas as pd

from scipy import misc
import matplotlib.pyplot as plt

IMAGE_SIZE = 64

def main():
    f = h5py.File('cats_vs_dogs_' + str(IMAGE_SIZE) + ".hdf5", "w")

    img_name_list = os.listdir('train')
    img_cnt       = len(img_name_list)
    img_idx       = np.random.permutation(img_cnt)

    y_train = np.array(['dog' in name for name in img_name_list], dtype=np.uint8)
    y_train = y_train[img_idx]
    f.create_dataset('train/y', (img_cnt,), dtype=np.uint8, data=y_train)

    dset = f.create_dataset('train/x', (img_cnt, IMAGE_SIZE, IMAGE_SIZE, 3), np.uint8)
    for i in img_idx:
            img = misc.imread(os.path.join('train', img_name_list[i]))
            arg_min, arg_max = (0,1) if img.shape[0] < img.shape[1] else (1,0)
            size = [IMAGE_SIZE] * 2
            size[arg_max] = img.shape[arg_max] * IMAGE_SIZE // img.shape[arg_min]
            y_min = (size[0] - IMAGE_SIZE) // 2
            x_min = (size[1] - IMAGE_SIZE) // 2
            dset[i] = misc.imresize(img, size)[y_min:y_min+IMAGE_SIZE,x_min:x_min+IMAGE_SIZE]


    img_name_list = os.listdir('test')
    img_cnt       = len(img_name_list)
    img_idx       = np.random.permutation(img_cnt)

    dset = f.create_dataset('test/x', (img_cnt, IMAGE_SIZE, IMAGE_SIZE, 3), np.uint8)
    for i in img_idx:
            img = misc.imread(os.path.join('test', img_name_list[i]))
            arg_min, arg_max = (0,1) if img.shape[0] < img.shape[1] else (1,0)
            size = [IMAGE_SIZE] * 2
            size[arg_max] = img.shape[arg_max] * IMAGE_SIZE // img.shape[arg_min]
            y_min = (size[0] - IMAGE_SIZE) // 2
            x_min = (size[1] - IMAGE_SIZE) // 2
            dset[i] = misc.imresize(img, size)[y_min:y_min+IMAGE_SIZE,x_min:x_min+IMAGE_SIZE]

    f.close()

if __name__ == '__main__':
    main()