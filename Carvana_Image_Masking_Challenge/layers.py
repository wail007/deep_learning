from keras.layers import Conv2D, BatchNormalization, concatenate
from keras.regularizers import l2
from keras.layers.advanced_activations import LeakyReLU


def conv_bn_relu(inputs, filter_cnt, filter_size=3, strides=1, wd=0.):
    layer = Conv2D(filter_cnt, filter_size, strides=strides, padding='same', kernel_initializer='he_uniform',
                   kernel_regularizer=l2(wd))(inputs)
    layer = BatchNormalization()(layer)
    layer = LeakyReLU()(layer)
    return layer


def dense_block(inputs, depth, growth_rate, wd=0.):
    layer_list = []
    for i in range(depth):
        layer = conv_bn_relu(inputs, growth_rate, wd=wd)
        inputs = concatenate([layer, inputs])
        layer_list.append(layer)
    return inputs, layer_list
