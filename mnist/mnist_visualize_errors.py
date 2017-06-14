import numpy  as np
import pandas as pd

from keras.datasets import mnist
from keras.models   import Sequential, load_model

from matplotlib import pyplot as plt

np.set_printoptions(precision=4, suppress=True, linewidth=200)

def main():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.expand_dims(x_train, axis=3).astype('float32') / 255
    x_test  = np.expand_dims(x_test , axis=3).astype('float32') / 255

    model = load_model(filepath='model_9956.hdf5')
    model.summary()

    prob = model.predict_proba(x_test, batch_size=1000)
    pred = prob.argmax(axis=1)

    wrong_pred = pred != y_test

    x_test_err = x_test[wrong_pred]
    y_test_err = y_test[wrong_pred]
    pred_err   = pred  [wrong_pred]
    prob_err   = prob  [wrong_pred]

    plt.gray()
    for i in range(len(pred_err)):
        print('(%d/%d)' % (i, len(pred_err)))
        print('Label      : %d' % y_test_err[i])
        print('Prediction : %d' % pred_err[i])
        print('Probability: ', end='')
        print(prob_err[i])

        plt.imshow(x_test_err[i,:,:,0])
        plt.show()

        



if __name__ == '__main__':
    main()