import numpy as np
import pandas as pd

from keras.models import Sequential, load_model
from keras.utils  import to_categorical

def main():
    train = pd.read_csv("train.csv", index_col=0).astype(np.float32)
    test  = pd.read_csv("test.csv" ).astype(np.float32)

    x_train = train.values / 255.0
    x_test  = test .values / 255.0
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test  = x_test .reshape(x_test .shape[0], 28, 28, 1)
    y_train = to_categorical(train.index.values, num_classes=10)

    model = load_model(filepath='model.hdf5')

    print(model.evaluate(x_train, y_train, batch_size=1000))

    pred = model.predict_classes(x_test, batch_size=1000)
    series = pd.Series(pred, np.arange(1, len(pred) + 1), name="Label")
    series.to_csv("submission.csv", header=True, index_label="ImageId")


if __name__ == '__main__':
    main()