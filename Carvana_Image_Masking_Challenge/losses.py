import tensorflow as tf
import keras.backend as k


def dice_loss(y_true, y_pred, logit=False):
    if logit:
        y_pred = k.sigmoid(y_pred)

    eps = k.epsilon()

    y_true = k.flatten(y_true)
    y_pred = k.flatten(y_pred)

    dice_coef = (2. * k.sum(y_true * y_pred) + eps) / (k.sum(y_true) + k.sum(y_pred) + eps)
    return 1. - dice_coef


def focal_loss(y_true, y_pred, focus=0.2):
    eps = k.epsilon()
    y_pred = k.clip(y_pred, eps, 1. - eps)
    return -k.mean(
        y_true * k.pow(1. - y_pred, focus) * k.log(y_pred) + (1. - y_true) * k.pow(y_pred, focus) * k.log(1. - y_pred))


def weighted_binary_crossentropy(y_true, y_pred, weight=1., logit=False):
    if not logit:
        eps = tf.convert_to_tensor(10e-8, dtype=y_pred.dtype.base_dtype)
        y_pred = tf.clip_by_value(y_pred, eps, 1 - eps)
        y_pred = tf.log(y_pred / (1 - y_pred))

    return k.mean(tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, weight), axis=-1)


def focal_dice_loss(y_true, y_pred, focus=0.2):
    return focal_loss(y_true, y_pred, focus) + dice_loss(y_true, y_pred)


def wbce_dice_loss(y_true, y_pred, weight=1.):
    return weighted_binary_crossentropy(y_true, y_pred, weight) + dice_loss(y_true, y_pred)
