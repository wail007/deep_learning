import tensorflow as tf
import keras.backend as K


def dice_coef(y_true, y_pred):
    threshold = 0.5
    smooth    = 1e-5

    y_true = K.cast(K.flatten(y_true) > threshold, dtype=K.floatx())
    y_pred = K.cast(K.flatten(y_pred) > threshold, dtype=K.floatx())

    return (2.0 * K.sum(y_true * y_pred) + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth)

def false_positive(y_true, y_pred):
   return K.sum(K.cast(tf.logical_and(K.flatten(y_true) <= 0.5, K.flatten(y_pred) > 0.5), dtype=K.floatx()))
    
def false_negative(y_true, y_pred):
   return K.sum(K.cast(tf.logical_and(K.flatten(y_true) > 0.5, K.flatten(y_pred) <= 0.5), dtype=K.floatx()))
