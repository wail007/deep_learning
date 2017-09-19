import tensorflow    as tf
import keras.backend as K

from keras.losses import binary_crossentropy


def dice_loss(y_true, y_pred, logit=False):
    if logit:
        y_pred = K.sigmoid(y_pred)

    smooth = 1e-8

    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)

    dice_coef =  (2. * K.sum(y_true * y_pred) + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth)

    return 1. - dice_coef

def weighted_binary_crossentropy(y_true, y_pred, weight=1., logit=False):
    # transform back to logits
    if not logit:
        epsilon = tf.convert_to_tensor(10e-8, dtype=y_pred.dtype.base_dtype)
        y_pred  = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
        y_pred  = tf.log(y_pred / (1 - y_pred))

    return K.mean(tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, weight), axis=-1)

def bce_dice_loss(y_true, y_pred):
    return weighted_binary_crossentropy(y_true, y_pred, 1.) + dice_loss(y_true, y_pred)


def wbce_loss(weights, y_true, y_pred):
    return (1. + weights) * binary_crossentropy(y_true, y_pred)


def wbce_dice_loss(weights, y_true, y_pred):
    return ((1. + weights) * binary_crossentropy(y_true, y_pred)) +  (1 - dice_coef(y_true, y_pred))
