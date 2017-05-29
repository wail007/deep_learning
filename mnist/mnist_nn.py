#!/usr/bin/env python
import numpy  as np
import pandas as pd

from keras.models                       import Sequential
from keras.layers                       import Dense, Activation, Lambda, Dropout
from keras.layers.noise                 import GaussianNoise, GaussianDropout
from keras.layers.advanced_activations  import ELU
from keras.optimizers                   import SGD 
from keras.regularizers                 import l1
from keras.callbacks                    import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image          import ImageDataGenerator
from keras.utils                        import to_categorical

def main():
    train = pd.read_csv("train.csv", index_col=0).astype(np.float32)
    test  = pd.read_csv("test.csv").astype(np.float32)

    x_train = train.values / 255.0
    x_test  = test .values / 255.0
    y_train = to_categorical(train.index.values, num_classes=10)

    model = Sequential()
    model.add(GaussianNoise(stddev=0.5, input_shape=(x_train.shape[1],)))
    #model.add(Dropout(rate=0.2, input_shape=(x_train.shape[1],)))
    #model.add(GaussianDropout(rate=0.3, input_shape=(x_train.shape[1],)))
    model.add(Dense(512, activation="relu"))#, input_dim=x_train.shape[1]))
    model.add(GaussianDropout(rate=0.5))
    model.add(Dense(128, activation="relu"))
    model.add(GaussianDropout(rate=0.5))
    model.add(Dense(10 , activation="softmax"))

    model.compile(optimizer=SGD(lr=0.5), 
                  loss="categorical_crossentropy", 
                  metrics=["accuracy"])

    chkpt = ModelCheckpoint(filepath="mnist_nn.hdf5", 
                            monitor="val_acc", 
                            verbose=1, 
                            save_best_only=True, 
                            save_weights_only=True,
                            mode="max")

    early_stop = EarlyStopping(monitor="val_acc", 
                               patience=5, 
                               verbose=1, 
                               mode="max")

    model.fit(x_train, y_train,
              batch_size=128, epochs=200, validation_split=0.3)

    #model.load_weights(filepath="mnist_nn.hdf5")

    pred = model.predict(x_test, batch_size=x_test.shape[0]).argmax(axis=1)

    series = pd.Series(pred, np.arange(1, len(pred) + 1), name="Label")
    series.to_csv("submission_relu.csv", header=True, index_label="ImageId")

if __name__ == '__main__':
    main()