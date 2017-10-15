import tensorflow as tf
import keras.backend as k


def dice_coef(y_true, y_pred):
    threshold = 0.5
    smooth = 1e-5

    y_true = k.cast(k.flatten(y_true) > threshold, dtype=k.floatx())
    y_pred = k.cast(k.flatten(y_pred) > threshold, dtype=k.floatx())

    return (2.0 * k.sum(y_true * y_pred) + smooth) / (k.sum(y_true) + k.sum(y_pred) + smooth)


def false_positive(y_true, y_pred):
    return k.sum(k.cast(tf.logical_and(k.flatten(y_true) <= 0.5, k.flatten(y_pred) > 0.5), dtype=k.floatx()))


def false_negative(y_true, y_pred):
    return k.sum(k.cast(tf.logical_and(k.flatten(y_true) > 0.5, k.flatten(y_pred) <= 0.5), dtype=k.floatx()))
