from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, BatchNormalization, UpSampling2D, ConvLSTM2D, \
    Conv2DTranspose
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import SGD
from keras.regularizers import l2

from losses import dice_loss, bce_dice_loss
from metrics import dice_coef, false_positive, false_negative
from layers import conv_bn_relu, dense_block


def get_unet(input_shape, pool_cnt, filter_cnt):
    inputs = Input(shape=input_shape, name='input')
    layer = inputs

    skip_connections = []
    for _ in range(pool_cnt):
        for _ in range(2):
            layer = conv_bn_relu(layer, filter_cnt)
        skip_connections.append(layer)
        layer = MaxPooling2D(2, strides=2)(layer)
        filter_cnt *= 2

    for _ in range(2):
        layer = conv_bn_relu(layer, filter_cnt)

    for _ in range(pool_cnt):
        filter_cnt //= 2
        layer = UpSampling2D(2)(layer)
        layer = concatenate([layer, skip_connections.pop()], axis=3)
        for _ in range(3):
            layer = conv_bn_relu(layer, filter_cnt)

    output = Conv2D(1, 1, activation='sigmoid', name='output')(layer)

    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=SGD(0.001, momentum=0.9, nesterov=True), loss=bce_dice_loss,
                  metrics=[dice_coef, false_positive, false_negative])
    model.summary()

    return model


def get_tiramisu(input_shape, depth_list, growth_rate_list, wd=0.):
    inputs = Input(shape=input_shape, name='input')
    layer = inputs

    skip_connections = []
    for depth, growth_rate in zip(depth_list[:-1], growth_rate_list[:-1]):
        layer, _ = dense_block(layer, depth, growth_rate, wd=wd)
        skip_connections.append(layer)
        layer = conv_bn_relu(layer, layer.get_shape().as_list()[-1], filter_size=1, strides=2, wd=wd)

    layer, layer_list = dense_block(layer, depth_list[-1], growth_rate_list[-1], wd=wd)

    for depth, growth_rate in zip(reversed(depth_list[:-1]), reversed(growth_rate_list[:-1])):
        layer = concatenate(layer_list)
        layer = UpSampling2D(2)(layer)
        layer = Conv2DTranspose(layer.get_shape().as_list()[-1], kernel_size=3, strides=2, padding='same',
                                kernel_initializer='he_uniform')(layer)
        layer, layer_list = dense_block(layer, depth, growth_rate, wd=wd)

    output = Conv2D(1, 1, activation='sigmoid', name='output', kernel_regularizer=l2(wd))(layer)

    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=SGD(0.005, momentum=0.9, nesterov=True), loss=bce_dice_loss,
                  metrics=[dice_coef, false_positive, false_negative])
    model.summary()

    return model


def get_unet_1024(input_shape=(1024, 1024, 3), num_classes=1):
    inputs = Input(shape=input_shape, name='input')
    # 1024

    down0b = Conv2D(8, 3, padding='same', kernel_initializer='he_uniform', name='down0b_conv1')(inputs)
    down0b = BatchNormalization(name='down0b_norm1')(down0b)
    down0b = LeakyReLU()(down0b)
    down0b = Conv2D(8, 3, padding='same', kernel_initializer='he_uniform', name='down0b_conv2')(down0b)
    down0b = BatchNormalization(name='down0b_norm2')(down0b)
    down0b = LeakyReLU()(down0b)
    down0b_pool = MaxPooling2D(2, strides=2)(down0b)
    # 512

    down0a = Conv2D(16, 3, padding='same', kernel_initializer='he_uniform', name='down0a_conv1')(down0b_pool)
    down0a = BatchNormalization(name='down0a_norm1')(down0a)
    down0a = LeakyReLU()(down0a)
    down0a = Conv2D(16, 3, padding='same', kernel_initializer='he_uniform', name='down0a_conv2')(down0a)
    down0a = BatchNormalization(name='down0a_norm2')(down0a)
    down0a = LeakyReLU()(down0a)
    down0a_pool = MaxPooling2D(2, strides=2)(down0a)
    # 256

    down0 = Conv2D(32, 3, padding='same', kernel_initializer='he_uniform', name='down0_conv1')(down0a_pool)
    down0 = BatchNormalization(name='down0_norm1')(down0)
    down0 = LeakyReLU()(down0)
    down0 = Conv2D(32, 3, padding='same', kernel_initializer='he_uniform', name='down0_conv2')(down0)
    down0 = BatchNormalization(name='down0_norm2')(down0)
    down0 = LeakyReLU()(down0)
    down0_pool = MaxPooling2D(2, strides=2)(down0)
    # 128

    down1 = Conv2D(64, 3, padding='same', kernel_initializer='he_uniform', name='down1_conv1')(down0_pool)
    down1 = BatchNormalization(name='down1_norm1')(down1)
    down1 = LeakyReLU()(down1)
    down1 = Conv2D(64, 3, padding='same', kernel_initializer='he_uniform', name='down1_conv2')(down1)
    down1 = BatchNormalization(name='down1_norm2')(down1)
    down1 = LeakyReLU()(down1)
    down1_pool = MaxPooling2D(2, strides=2)(down1)
    # 64

    down2 = Conv2D(128, 3, padding='same', kernel_initializer='he_uniform', name='down2_conv1')(down1_pool)
    down2 = BatchNormalization(name='down2_norm1')(down2)
    down2 = LeakyReLU()(down2)
    down2 = Conv2D(128, 3, padding='same', kernel_initializer='he_uniform', name='down2_conv2')(down2)
    down2 = BatchNormalization(name='down2_norm2')(down2)
    down2 = LeakyReLU()(down2)
    down2_pool = MaxPooling2D(2, strides=2)(down2)
    # 32

    down3 = Conv2D(256, 3, padding='same', kernel_initializer='he_uniform', name='down3_conv1')(down2_pool)
    down3 = BatchNormalization(name='down3_norm1')(down3)
    down3 = LeakyReLU()(down3)
    down3 = Conv2D(256, 3, padding='same', kernel_initializer='he_uniform', name='down3_conv2')(down3)
    down3 = BatchNormalization(name='down3_norm2')(down3)
    down3 = LeakyReLU()(down3)
    down3_pool = MaxPooling2D(2, strides=2)(down3)
    # 16

    down4 = Conv2D(512, 3, padding='same', kernel_initializer='he_uniform', name='down4_conv1')(down3_pool)
    down4 = BatchNormalization(name='down4_norm1')(down4)
    down4 = LeakyReLU()(down4)
    down4 = Conv2D(512, 3, padding='same', kernel_initializer='he_uniform', name='down4_conv2')(down4)
    down4 = BatchNormalization(name='down4_norm2')(down4)
    down4 = LeakyReLU()(down4)
    down4_pool = MaxPooling2D(2, strides=2)(down4)
    # 8

    center = Conv2D(1024, 3, padding='same', kernel_initializer='he_uniform', name='center_conv1')(down4_pool)
    center = BatchNormalization(name='center_norm1')(center)
    center = LeakyReLU()(center)
    center = Conv2D(1024, 3, padding='same', kernel_initializer='he_uniform', name='center_conv2')(center)
    center = BatchNormalization(name='center_norm2')(center)
    center = LeakyReLU()(center)
    # center

    up4 = UpSampling2D(2)(center)
    up4 = concatenate([down4, up4], axis=3)
    up4 = Conv2D(512, 3, padding='same', kernel_initializer='he_uniform', name='up4_conv1')(up4)
    up4 = BatchNormalization(name='up4_norm1')(up4)
    up4 = LeakyReLU()(up4)
    up4 = Conv2D(512, 3, padding='same', kernel_initializer='he_uniform', name='up4_conv2')(up4)
    up4 = BatchNormalization(name='up4_norm2')(up4)
    up4 = LeakyReLU()(up4)
    up4 = Conv2D(512, 3, padding='same', kernel_initializer='he_uniform', name='up4_conv3')(up4)
    up4 = BatchNormalization(name='up4_norm3')(up4)
    up4 = LeakyReLU()(up4)
    # 16

    up3 = UpSampling2D(2)(up4)
    up3 = concatenate([down3, up3], axis=3)
    up3 = Conv2D(256, 3, padding='same', kernel_initializer='he_uniform', name='up3_conv1')(up3)
    up3 = BatchNormalization(name='up3_norm1')(up3)
    up3 = LeakyReLU()(up3)
    up3 = Conv2D(256, 3, padding='same', kernel_initializer='he_uniform', name='up3_conv2')(up3)
    up3 = BatchNormalization(name='up3_norm2')(up3)
    up3 = LeakyReLU()(up3)
    up3 = Conv2D(256, 3, padding='same', kernel_initializer='he_uniform', name='up3_conv3')(up3)
    up3 = BatchNormalization(name='up3_norm3')(up3)
    up3 = LeakyReLU()(up3)
    # 32

    up2 = UpSampling2D(2)(up3)
    up2 = concatenate([down2, up2], axis=3)
    up2 = Conv2D(128, 3, padding='same', kernel_initializer='he_uniform', name='up2_conv1')(up2)
    up2 = BatchNormalization(name='up2_norm1')(up2)
    up2 = LeakyReLU()(up2)
    up2 = Conv2D(128, 3, padding='same', kernel_initializer='he_uniform', name='up2_conv2')(up2)
    up2 = BatchNormalization(name='up2_norm2')(up2)
    up2 = LeakyReLU()(up2)
    up2 = Conv2D(128, 3, padding='same', kernel_initializer='he_uniform', name='up2_conv3')(up2)
    up2 = BatchNormalization(name='up2_norm3')(up2)
    up2 = LeakyReLU()(up2)
    # 64

    up1 = UpSampling2D(2)(up2)
    up1 = concatenate([down1, up1], axis=3)
    up1 = Conv2D(64, 3, padding='same', kernel_initializer='he_uniform', name='up1_conv1')(up1)
    up1 = BatchNormalization(name='up1_norm1')(up1)
    up1 = LeakyReLU()(up1)
    up1 = Conv2D(64, 3, padding='same', kernel_initializer='he_uniform', name='up1_conv2')(up1)
    up1 = BatchNormalization(name='up1_norm2')(up1)
    up1 = LeakyReLU()(up1)
    up1 = Conv2D(64, 3, padding='same', kernel_initializer='he_uniform', name='up1_conv3')(up1)
    up1 = BatchNormalization(name='up1_norm3')(up1)
    up1 = LeakyReLU()(up1)
    # 128

    up0 = UpSampling2D(2)(up1)
    up0 = concatenate([down0, up0], axis=3)
    up0 = Conv2D(32, 3, padding='same', kernel_initializer='he_uniform', name='up0_conv1')(up0)
    up0 = BatchNormalization(name='up0_norm1')(up0)
    up0 = LeakyReLU()(up0)
    up0 = Conv2D(32, 3, padding='same', kernel_initializer='he_uniform', name='up0_conv2')(up0)
    up0 = BatchNormalization(name='up0_norm2')(up0)
    up0 = LeakyReLU()(up0)
    up0 = Conv2D(32, 3, padding='same', kernel_initializer='he_uniform', name='up0_conv3')(up0)
    up0 = BatchNormalization(name='up0_norm3')(up0)
    up0 = LeakyReLU()(up0)
    # 256

    up0a = UpSampling2D(2)(up0)
    up0a = concatenate([down0a, up0a], axis=3)
    up0a = Conv2D(16, 3, padding='same', kernel_initializer='he_uniform', name='up0a_conv1')(up0a)
    up0a = BatchNormalization(name='up0a_norm1')(up0a)
    up0a = LeakyReLU()(up0a)
    up0a = Conv2D(16, 3, padding='same', kernel_initializer='he_uniform', name='up0a_conv2')(up0a)
    up0a = BatchNormalization(name='up0a_norm2')(up0a)
    up0a = LeakyReLU()(up0a)
    up0a = Conv2D(16, 3, padding='same', kernel_initializer='he_uniform', name='up0a_conv3')(up0a)
    up0a = BatchNormalization(name='up0a_norm3')(up0a)
    up0a = LeakyReLU()(up0a)
    # 512

    up0b = UpSampling2D(2)(up0a)
    up0b = concatenate([down0b, up0b], axis=3)
    up0b = Conv2D(8, 3, padding='same', kernel_initializer='he_uniform', name='up0b_conv1')(up0b)
    up0b = BatchNormalization(name='up0b_norm1')(up0b)
    up0b = LeakyReLU()(up0b)
    up0b = Conv2D(8, 3, padding='same', kernel_initializer='he_uniform', name='up0b_conv2')(up0b)
    up0b = BatchNormalization(name='up0b_norm2')(up0b)
    up0b = LeakyReLU()(up0b)
    up0b = Conv2D(8, 3, padding='same', kernel_initializer='he_uniform', name='up0b_conv3')(up0b)
    up0b = BatchNormalization(name='up0b_norm3')(up0b)
    up0b = LeakyReLU()(up0b)
    # 1024

    classify = Conv2D(num_classes, 1, activation='sigmoid', name='output')(up0b)

    model = Model(inputs=inputs, outputs=classify)
    model.compile(optimizer=SGD(0.01, momentum=0.9, nesterov=True), loss=bce_dice_loss,
                  metrics=[dice_coef, false_positive, false_negative])
    model.summary()

    return model


def get_seq_unet_1024(input_shape=(4, 1024, 1024, 3), num_classes=1, unet_trainable=True):
    inputs = Input(shape=input_shape, name='input')
    # 1024

    down0b = TimeDistributed(Conv2D(8, 3, padding='same', kernel_initializer='he_normal', trainable=unet_trainable),
                             name='down0b_conv1')(inputs)
    down0b = TimeDistributed(BatchNormalization(trainable=unet_trainable), name='down0b_norm1')(down0b)
    down0b = TimeDistributed(LeakyReLU())(down0b)
    down0b = TimeDistributed(Conv2D(8, 3, padding='same', kernel_initializer='he_normal', trainable=unet_trainable),
                             name='down0b_conv2')(down0b)
    down0b = TimeDistributed(BatchNormalization(trainable=unet_trainable), name='down0b_norm2')(down0b)
    down0b = TimeDistributed(LeakyReLU())(down0b)
    down0b_pool = TimeDistributed(MaxPooling2D(2, strides=2))(down0b)
    # 512

    down0a = TimeDistributed(Conv2D(16, 3, padding='same', kernel_initializer='he_normal', trainable=unet_trainable),
                             name='down0a_conv1')(down0b_pool)
    down0a = TimeDistributed(BatchNormalization(trainable=unet_trainable), name='down0a_norm1')(down0a)
    down0a = TimeDistributed(LeakyReLU())(down0a)
    down0a = TimeDistributed(Conv2D(16, 3, padding='same', kernel_initializer='he_normal', trainable=unet_trainable),
                             name='down0a_conv2')(down0a)
    down0a = TimeDistributed(BatchNormalization(trainable=unet_trainable), name='down0a_norm2')(down0a)
    down0a = TimeDistributed(LeakyReLU())(down0a)
    down0a_pool = TimeDistributed(MaxPooling2D(2, strides=2))(down0a)
    # 256

    down0 = TimeDistributed(Conv2D(32, 3, padding='same', kernel_initializer='he_normal', trainable=unet_trainable),
                            name='down0_conv1')(down0a_pool)
    down0 = TimeDistributed(BatchNormalization(trainable=unet_trainable), name='down0_norm1')(down0)
    down0 = TimeDistributed(LeakyReLU())(down0)
    down0 = TimeDistributed(Conv2D(32, 3, padding='same', kernel_initializer='he_normal', trainable=unet_trainable),
                            name='down0_conv2')(down0)
    down0 = TimeDistributed(BatchNormalization(trainable=unet_trainable), name='down0_norm2')(down0)
    down0 = TimeDistributed(LeakyReLU())(down0)
    down0_pool = TimeDistributed(MaxPooling2D(2, strides=2))(down0)
    # 128

    down1 = TimeDistributed(Conv2D(64, 3, padding='same', kernel_initializer='he_normal', trainable=unet_trainable),
                            name='down1_conv1')(down0_pool)
    down1 = TimeDistributed(BatchNormalization(trainable=unet_trainable), name='down1_norm1')(down1)
    down1 = TimeDistributed(LeakyReLU())(down1)
    down1 = TimeDistributed(Conv2D(64, 3, padding='same', kernel_initializer='he_normal', trainable=unet_trainable),
                            name='down1_conv2')(down1)
    down1 = TimeDistributed(BatchNormalization(trainable=unet_trainable), name='down1_norm2')(down1)
    down1 = TimeDistributed(LeakyReLU())(down1)
    down1_pool = TimeDistributed(MaxPooling2D(2, strides=2))(down1)
    # 64

    down2 = TimeDistributed(Conv2D(128, 3, padding='same', kernel_initializer='he_normal', trainable=unet_trainable),
                            name='down2_conv1')(down1_pool)
    down2 = TimeDistributed(BatchNormalization(trainable=unet_trainable), name='down2_norm1')(down2)
    down2 = TimeDistributed(LeakyReLU())(down2)
    down2 = TimeDistributed(Conv2D(128, 3, padding='same', kernel_initializer='he_normal', trainable=unet_trainable),
                            name='down2_conv2')(down2)
    down2 = TimeDistributed(BatchNormalization(trainable=unet_trainable), name='down2_norm2')(down2)
    down2 = TimeDistributed(LeakyReLU())(down2)
    down2_pool = TimeDistributed(MaxPooling2D(2, strides=2))(down2)
    # 32

    down3 = TimeDistributed(Conv2D(256, 3, padding='same', kernel_initializer='he_normal', trainable=unet_trainable),
                            name='down3_conv1')(down2_pool)
    down3 = TimeDistributed(BatchNormalization(trainable=unet_trainable), name='down3_norm1')(down3)
    down3 = TimeDistributed(LeakyReLU())(down3)
    down3 = TimeDistributed(Conv2D(256, 3, padding='same', kernel_initializer='he_normal', trainable=unet_trainable),
                            name='down3_conv2')(down3)
    down3 = TimeDistributed(BatchNormalization(trainable=unet_trainable), name='down3_norm2')(down3)
    down3 = TimeDistributed(LeakyReLU())(down3)
    down3_pool = TimeDistributed(MaxPooling2D(2, strides=2))(down3)
    # 16

    down4 = TimeDistributed(Conv2D(512, 3, padding='same', kernel_initializer='he_normal', trainable=unet_trainable),
                            name='down4_conv1')(down3_pool)
    down4 = TimeDistributed(BatchNormalization(trainable=unet_trainable), name='down4_norm1')(down4)
    down4 = TimeDistributed(LeakyReLU())(down4)
    down4 = TimeDistributed(Conv2D(512, 3, padding='same', kernel_initializer='he_normal', trainable=unet_trainable),
                            name='down4_conv2')(down4)
    down4 = TimeDistributed(BatchNormalization(trainable=unet_trainable), name='down4_norm2')(down4)
    down4 = TimeDistributed(LeakyReLU())(down4)
    down4_pool = TimeDistributed(MaxPooling2D(2, strides=2))(down4)
    # 8

    center = TimeDistributed(Conv2D(1024, 3, padding='same', kernel_initializer='he_normal', trainable=unet_trainable),
                             name='center_conv1')(down4_pool)
    center = TimeDistributed(BatchNormalization(trainable=unet_trainable), name='center_norm1')(center)
    center = TimeDistributed(LeakyReLU())(center)
    center = TimeDistributed(Conv2D(1024, 3, padding='same', kernel_initializer='he_normal', trainable=unet_trainable),
                             name='center_conv2')(center)
    center = TimeDistributed(BatchNormalization(trainable=unet_trainable), name='center_norm2')(center)
    center = TimeDistributed(LeakyReLU())(center)
    # center

    up4 = TimeDistributed(UpSampling2D(2))(center)
    up4 = concatenate([down4, up4], axis=4)
    up4 = TimeDistributed(Conv2D(512, 3, padding='same', kernel_initializer='he_normal', trainable=unet_trainable),
                          name='up4_conv1')(up4)
    up4 = TimeDistributed(BatchNormalization(trainable=unet_trainable), name='up4_norm1')(up4)
    up4 = TimeDistributed(LeakyReLU())(up4)
    up4 = TimeDistributed(Conv2D(512, 3, padding='same', kernel_initializer='he_normal', trainable=unet_trainable),
                          name='up4_conv2')(up4)
    up4 = TimeDistributed(BatchNormalization(trainable=unet_trainable), name='up4_norm2')(up4)
    up4 = TimeDistributed(LeakyReLU())(up4)
    up4 = TimeDistributed(Conv2D(512, 3, padding='same', kernel_initializer='he_normal', trainable=unet_trainable),
                          name='up4_conv3')(up4)
    up4 = TimeDistributed(BatchNormalization(trainable=unet_trainable), name='up4_norm3')(up4)
    up4 = TimeDistributed(LeakyReLU())(up4)
    # 16

    up3 = TimeDistributed(UpSampling2D(2))(up4)
    up3 = concatenate([down3, up3], axis=4)
    up3 = TimeDistributed(Conv2D(256, 3, padding='same', kernel_initializer='he_normal', trainable=unet_trainable),
                          name='up3_conv1')(up3)
    up3 = TimeDistributed(BatchNormalization(trainable=unet_trainable), name='up3_norm1')(up3)
    up3 = TimeDistributed(LeakyReLU())(up3)
    up3 = TimeDistributed(Conv2D(256, 3, padding='same', kernel_initializer='he_normal', trainable=unet_trainable),
                          name='up3_conv2')(up3)
    up3 = TimeDistributed(BatchNormalization(trainable=unet_trainable), name='up3_norm2')(up3)
    up3 = TimeDistributed(LeakyReLU())(up3)
    up3 = TimeDistributed(Conv2D(256, 3, padding='same', kernel_initializer='he_normal', trainable=unet_trainable),
                          name='up3_conv3')(up3)
    up3 = TimeDistributed(BatchNormalization(trainable=unet_trainable), name='up3_norm3')(up3)
    up3 = TimeDistributed(LeakyReLU())(up3)
    # 32

    up2 = TimeDistributed(UpSampling2D(2))(up3)
    up2 = concatenate([down2, up2], axis=4)
    up2 = TimeDistributed(Conv2D(128, 3, padding='same', kernel_initializer='he_normal', trainable=unet_trainable),
                          name='up2_conv1')(up2)
    up2 = TimeDistributed(BatchNormalization(trainable=unet_trainable), name='up2_norm1')(up2)
    up2 = TimeDistributed(LeakyReLU())(up2)
    up2 = TimeDistributed(Conv2D(128, 3, padding='same', kernel_initializer='he_normal', trainable=unet_trainable),
                          name='up2_conv2')(up2)
    up2 = TimeDistributed(BatchNormalization(trainable=unet_trainable), name='up2_norm2')(up2)
    up2 = TimeDistributed(LeakyReLU())(up2)
    up2 = TimeDistributed(Conv2D(128, 3, padding='same', kernel_initializer='he_normal', trainable=unet_trainable),
                          name='up2_conv3')(up2)
    up2 = TimeDistributed(BatchNormalization(trainable=unet_trainable), name='up2_norm3')(up2)
    up2 = TimeDistributed(LeakyReLU())(up2)
    # 64

    up1 = TimeDistributed(UpSampling2D(2))(up2)
    up1 = concatenate([down1, up1], axis=4)
    up1 = TimeDistributed(Conv2D(64, 3, padding='same', kernel_initializer='he_normal', trainable=unet_trainable),
                          name='up1_conv1')(up1)
    up1 = TimeDistributed(BatchNormalization(trainable=unet_trainable), name='up1_norm1')(up1)
    up1 = TimeDistributed(LeakyReLU())(up1)
    up1 = TimeDistributed(Conv2D(64, 3, padding='same', kernel_initializer='he_normal', trainable=unet_trainable),
                          name='up1_conv2')(up1)
    up1 = TimeDistributed(BatchNormalization(trainable=unet_trainable), name='up1_norm2')(up1)
    up1 = TimeDistributed(LeakyReLU())(up1)
    up1 = TimeDistributed(Conv2D(64, 3, padding='same', kernel_initializer='he_normal', trainable=unet_trainable),
                          name='up1_conv3')(up1)
    up1 = TimeDistributed(BatchNormalization(trainable=unet_trainable), name='up1_norm3')(up1)
    up1 = TimeDistributed(LeakyReLU())(up1)
    # 128

    up0 = TimeDistributed(UpSampling2D(2))(up1)
    up0 = concatenate([down0, up0], axis=4)
    up0 = TimeDistributed(Conv2D(32, 3, padding='same', kernel_initializer='he_normal', trainable=unet_trainable),
                          name='up0_conv1')(up0)
    up0 = TimeDistributed(BatchNormalization(trainable=unet_trainable), name='up0_norm1')(up0)
    up0 = TimeDistributed(LeakyReLU())(up0)
    up0 = TimeDistributed(Conv2D(32, 3, padding='same', kernel_initializer='he_normal', trainable=unet_trainable),
                          name='up0_conv2')(up0)
    up0 = TimeDistributed(BatchNormalization(trainable=unet_trainable), name='up0_norm2')(up0)
    up0 = TimeDistributed(LeakyReLU())(up0)
    up0 = TimeDistributed(Conv2D(32, 3, padding='same', kernel_initializer='he_normal', trainable=unet_trainable),
                          name='up0_conv3')(up0)
    up0 = TimeDistributed(BatchNormalization(trainable=unet_trainable), name='up0_norm3')(up0)
    up0 = TimeDistributed(LeakyReLU())(up0)
    # 256

    up0a = TimeDistributed(UpSampling2D(2))(up0)
    up0a = concatenate([down0a, up0a], axis=4)
    up0a = TimeDistributed(Conv2D(16, 3, padding='same', kernel_initializer='he_normal', trainable=unet_trainable),
                           name='up0a_conv1')(up0a)
    up0a = TimeDistributed(BatchNormalization(trainable=unet_trainable), name='up0a_norm1')(up0a)
    up0a = TimeDistributed(LeakyReLU())(up0a)
    up0a = TimeDistributed(Conv2D(16, 3, padding='same', kernel_initializer='he_normal', trainable=unet_trainable),
                           name='up0a_conv2')(up0a)
    up0a = TimeDistributed(BatchNormalization(trainable=unet_trainable), name='up0a_norm2')(up0a)
    up0a = TimeDistributed(LeakyReLU())(up0a)
    up0a = TimeDistributed(Conv2D(16, 3, padding='same', kernel_initializer='he_normal', trainable=unet_trainable),
                           name='up0a_conv3')(up0a)
    up0a = TimeDistributed(BatchNormalization(trainable=unet_trainable), name='up0a_norm3')(up0a)
    up0a = TimeDistributed(LeakyReLU())(up0a)
    # 512

    up0b = TimeDistributed(UpSampling2D(2))(up0a)
    up0b = concatenate([down0b, up0b], axis=4)
    up0b = TimeDistributed(Conv2D(8, 3, padding='same', kernel_initializer='he_normal', trainable=unet_trainable),
                           name='up0b_conv1')(up0b)
    up0b = TimeDistributed(BatchNormalization(trainable=unet_trainable), name='up0b_norm1')(up0b)
    up0b = TimeDistributed(LeakyReLU())(up0b)
    up0b = TimeDistributed(Conv2D(8, 3, padding='same', kernel_initializer='he_normal', trainable=unet_trainable),
                           name='up0b_conv2')(up0b)
    up0b = TimeDistributed(BatchNormalization(trainable=unet_trainable), name='up0b_norm2')(up0b)
    up0b = TimeDistributed(LeakyReLU())(up0b)
    up0b = TimeDistributed(Conv2D(8, 3, padding='same', kernel_initializer='he_normal', trainable=unet_trainable),
                           name='up0b_conv3')(up0b)
    up0b = TimeDistributed(BatchNormalization(trainable=unet_trainable), name='up0b_norm3')(up0b)
    up0b = TimeDistributed(LeakyReLU())(up0b)
    # 1024

    classify = Bidirectional(
        ConvLSTM2D(4, 3, padding='same', activation=None, kernel_initializer='he_normal', return_sequences=True))(up0b)
    classify = TimeDistributed(BatchNormalization())(classify)
    classify = TimeDistributed(LeakyReLU())(classify)

    classify = TimeDistributed(Conv2D(num_classes, 1, activation='sigmoid'))(classify)

    model = Model(inputs=inputs, outputs=classify)
    model.compile(optimizer=SGD(0.001, momentum=0.9, nesterov=True), loss=bce_dice_loss,
                  metrics=[dice_coef, false_positive, false_negative])
    model.summary()

    return model
