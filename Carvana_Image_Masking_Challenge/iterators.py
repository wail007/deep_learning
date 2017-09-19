import numpy as np

from keras.preprocessing.image  import Iterator


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
            batch_x = self.x[index_array[0]:index_array[-1] + 1]
            batch_y = self.y[index_array[0]:index_array[-1] + 1]

        if self.preprocess is not None:
            batch_x, batch_y = self.preprocess(batch_x, batch_y)

        return batch_x, batch_y

