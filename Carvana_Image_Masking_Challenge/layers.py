from keras.layers import Conv2D, Activation, BatchNormalization, Dropout, MaxPool2D, concatenate, Conv2DTranspose
from keras.regularizers import l2

def bn_relu_conv(inputs, n_filters, filter_size=3, dropout_p=0.2, wd=0.):
    layer = BatchNormalization()(inputs)
    layer = Activation('relu')(layer)
    layer = Conv2D(n_filters, filter_size, padding='same', kernel_initializer='he_uniform', kernel_regularizer=l2(wd))(layer)

    if dropout_p != 0.0:
        layer = Dropout(dropout_p)(layer)

    return layer


def dense_block(inputs, depth, growth_rate, dropout_p=0.2, wd=0.):
    layer_list = []
    for i in range(depth):
        layer  = bn_relu_conv(inputs, growth_rate, dropout_p=dropout_p, wd=wd)
        inputs = concatenate([inputs, layer])
        layer_list.append(layer)
    return inputs, layer_list


def transition_down(inputs, n_filters, dropout_p=0.2, wd=0.):
    layer = bn_relu_conv(inputs, n_filters if n_filters else inputs.get_shape().as_list()[-1], filter_size=1, dropout_p=dropout_p, wd=wd)
    return MaxPool2D(2, strides=2)(layer)


def down_path(inputs, depth_list, growth_rate_list, n_filters_list, dropout_p=0.2, wd=0.):
    skip_connections = []
    for depth, growth_rate, n_filters in zip(depth_list, growth_rate_list, n_filters_list):
        inputs, _ = dense_block(inputs, depth, growth_rate, dropout_p, wd)
        skip_connections.append(inputs)
        inputs = transition_down(inputs, n_filters, dropout_p, wd)
    return inputs, skip_connections


def transition_up(layer_list, n_filters, skip_connection, wd=0.):
    layer = concatenate(layer_list)
    layer = Conv2DTranspose(n_filters if n_filters else layer.get_shape().as_list()[-1], kernel_size=3, strides=2, padding='same', kernel_initializer='he_uniform', kernel_regularizer=l2(wd))(layer)
    return concatenate([layer, skip_connection])


def up_path(layer_list, skip_connections, depth_list, growth_rate_list, n_filters_list, dropout_p=0.2, wd=0.):
    for depth, growth_rate, skip_connection, n_filters in zip(depth_list, growth_rate_list, skip_connections, n_filters_list):
        layer = transition_up(layer_list, n_filters, skip_connection, wd)
        layer, layer_list  = dense_block(layer, depth, growth_rate, dropout_p, wd)
    return layer
