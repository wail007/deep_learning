#!/usr/bin/env python
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pdb
import argparse
import urllib.request
import io

import numpy                as np
import matplotlib.pyplot    as plt

from scipy          import misc
from keras          import backend as K
from keras.models   import load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', required=True)
    parser.add_argument('-i', '--input', required=True)
    args = parser.parse_args()

    model = load_model(args.model)

    if 'http' in args.input:
        with urllib.request.urlopen(args.input) as url:
            args.input = io.BytesIO(url.read())

    img = misc.imread(args.input)
    
    img_input = misc.imresize(img, model.layers[0].input_shape[1:3])
    img_input = img_input.astype(K.floatx()) / 255
    img_input = np.expand_dims(img_input, axis=0)

    prob  = model.predict_on_batch(img_input)

    if prob < 0.5:
        print('Cat with %.2f%% probability' % ((1.0-prob)*100,))
    else:
        print('Dog with %.2f%% probability' % (prob*100.0,))

    plt.imshow(img)
    plt.show()



if __name__ == '__main__':
    main()