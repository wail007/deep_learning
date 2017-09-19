import pdb
from functools import partial, update_wrapper

from keras.models                       import Model
from keras.layers                       import Input, Conv2D, MaxPooling2D, Activation, concatenate, BatchNormalization, UpSampling2D, ConvLSTM2D, Convolution2D, Dropout, merge, Deconvolution2D
from keras.layers.wrappers              import TimeDistributed, Bidirectional
from keras.layers.advanced_activations  import LeakyReLU
from keras.optimizers                   import SGD, RMSprop, Adam, Nadam
from keras.regularizers                 import l2

from losses  import bce_dice_loss
from metrics import dice_coef, false_positive, false_negative
from layers import dense_block, down_path, up_path



def get_unet_128(input_shape=(128, 128, 3),
                 num_classes=1):
    inputs = Input(shape=input_shape)
    #inputs_noise = GaussianNoise(0.01)(inputs)
    # 128

    down1 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(inputs)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(down1)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
    # 64

    down2 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(down1_pool)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(down2)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)
    # 32

    down3 = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal')(down2_pool)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3 = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal')(down3)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)
    # 16

    down4 = Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal')(down3_pool)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4 = Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal')(down4)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)
    # 8

    center = Conv2D(1024, (3, 3), padding='same', kernel_initializer='he_normal')(down4_pool)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    center = Conv2D(1024, (3, 3), padding='same', kernel_initializer='he_normal')(center)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    # center

    up4 = UpSampling2D((2, 2))(center)
    up4 = concatenate([down4, up4], axis=3)
    up4 = Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    # 16

    up3 = UpSampling2D((2, 2))(up4)
    up3 = concatenate([down3, up3], axis=3)
    up3 = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    # 32

    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([down2, up2], axis=3)
    up2 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    # 64

    up1 = UpSampling2D((2, 2))(up2)
    up1 = concatenate([down1, up1], axis=3)
    up1 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    # 128

    classify = Conv2D(num_classes, (1, 1), activation='sigmoid')(up1)

    model = Model(inputs=inputs, outputs=classify)

    model.compile(optimizer=SGD(lr=0.1), loss=bce_dice_loss, metrics=[dice_coef])

    return model


def get_lstm_unet_128(input_shape=(16, 128, 128, 3), num_classes=1):
    inputs = Input(shape=input_shape)
    #inputs_noise = GaussianNoise(0.01)(inputs)
    # 128

    down1 = TimeDistributed(Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal'))(inputs)
    down1 = TimeDistributed(BatchNormalization())(down1)
    down1 = TimeDistributed(LeakyReLU())(down1)
    down1 = TimeDistributed(Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal'))(down1)
    down1 = TimeDistributed(BatchNormalization())(down1)
    down1 = TimeDistributed(LeakyReLU())(down1)
    down1_pool = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2)))(down1)
    # 64

    down2 = TimeDistributed(Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal'))(down1_pool)
    down2 = TimeDistributed(BatchNormalization())(down2)
    down2 = TimeDistributed(LeakyReLU())(down2)
    down2 = TimeDistributed(Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal'))(down2)
    down2 = TimeDistributed(BatchNormalization())(down2)
    down2 = TimeDistributed(LeakyReLU())(down2)
    down2_pool = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2)))(down2)
    # 32

    down3 = TimeDistributed(Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal'))(down2_pool)
    down3 = TimeDistributed(BatchNormalization())(down3)
    down3 = TimeDistributed(LeakyReLU())(down3)
    down3 = TimeDistributed(Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal'))(down3)
    down3 = TimeDistributed(BatchNormalization())(down3)
    down3 = TimeDistributed(LeakyReLU())(down3)
    down3_pool = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2)))(down3)
    # 16

    down4 = TimeDistributed(Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal'))(down3_pool)
    down4 = TimeDistributed(BatchNormalization())(down4)
    down4 = TimeDistributed(LeakyReLU())(down4)
    down4 = TimeDistributed(Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal'))(down4)
    down4 = TimeDistributed(BatchNormalization())(down4)
    down4 = TimeDistributed(LeakyReLU())(down4)
    down4_pool = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2)))(down4)
    # 8

    center = TimeDistributed(Conv2D(1024, (3, 3), padding='same', kernel_initializer='he_normal'))(down4_pool)
    center = TimeDistributed(BatchNormalization())(center)
    center = TimeDistributed(LeakyReLU())(center)
    #center = Bidirectional(ConvLSTM2D(20, 3, padding='same', activation=None, kernel_initializer='he_normal', return_sequences=True))(center)
    #center = TimeDistributed(BatchNormalization())(center)
    #center = TimeDistributed(LeakyReLU())(center)
    center = TimeDistributed(Conv2D(1024, (3, 3), padding='same', kernel_initializer='he_normal'))(center)
    center = TimeDistributed(BatchNormalization())(center)
    center = TimeDistributed(LeakyReLU())(center)
    # center

    up4 = TimeDistributed(UpSampling2D((2, 2)))(center)
    up4 = concatenate([down4, up4], axis=4)
    up4 = TimeDistributed(Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal'))(up4)
    up4 = TimeDistributed(BatchNormalization())(up4)
    up4 = TimeDistributed(LeakyReLU())(up4)
    up4 = TimeDistributed(Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal'))(up4)
    up4 = TimeDistributed(BatchNormalization())(up4)
    up4 = TimeDistributed(LeakyReLU())(up4)
    up4 = TimeDistributed(Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal'))(up4)
    up4 = TimeDistributed(BatchNormalization())(up4)
    up4 = TimeDistributed(LeakyReLU())(up4)
    # 16

    up3 = TimeDistributed(UpSampling2D((2, 2)))(up4)
    up3 = concatenate([down3, up3], axis=4)
    up3 = TimeDistributed(Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal'))(up3)
    up3 = TimeDistributed(BatchNormalization())(up3)
    up3 = TimeDistributed(LeakyReLU())(up3)
    up3 = TimeDistributed(Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal'))(up3)
    up3 = TimeDistributed(BatchNormalization())(up3)
    up3 = TimeDistributed(LeakyReLU())(up3)
    up3 = TimeDistributed(Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal'))(up3)
    up3 = TimeDistributed(BatchNormalization())(up3)
    up3 = TimeDistributed(LeakyReLU())(up3)
    # 32

    up2 = TimeDistributed(UpSampling2D((2, 2)))(up3)
    up2 = concatenate([down2, up2], axis=4)
    up2 = TimeDistributed(Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal'))(up2)
    up2 = TimeDistributed(BatchNormalization())(up2)
    up2 = TimeDistributed(LeakyReLU())(up2)
    up2 = TimeDistributed(Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal'))(up2)
    up2 = TimeDistributed(BatchNormalization())(up2)
    up2 = TimeDistributed(LeakyReLU())(up2)
    up2 = TimeDistributed(Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal'))(up2)
    up2 = TimeDistributed(BatchNormalization())(up2)
    up2 = TimeDistributed(LeakyReLU())(up2)
    # 64

    up1 = TimeDistributed(UpSampling2D((2, 2)))(up2)
    up1 = concatenate([down1, up1], axis=4)
    up1 = TimeDistributed(Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal'))(up1)
    up1 = TimeDistributed(BatchNormalization())(up1)
    up1 = TimeDistributed(LeakyReLU())(up1)
    up1 = TimeDistributed(Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal'))(up1)
    up1 = TimeDistributed(BatchNormalization())(up1)
    up1 = TimeDistributed(LeakyReLU())(up1)
    up1 = TimeDistributed(Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal'))(up1)
    up1 = TimeDistributed(BatchNormalization())(up1)
    up1 = TimeDistributed(LeakyReLU())(up1)
    # 128

    classify = TimeDistributed(Conv2D(num_classes, (1, 1), activation='sigmoid'))(up1)

    model = Model(inputs=inputs, outputs=classify)
    model.compile(optimizer=SGD(lr=0.01, momentum=0.99, nesterov=True), loss=bce_dice_loss, metrics=[dice_coef])
    model.summary()

    return model


def get_unet_512(input_shape=(512, 512, 3), num_classes=1):

    inputs = Input(shape=input_shape, name='input')
    # 512 

    down0a = Conv2D(16, 3, padding='same', kernel_initializer='he_normal', name='down0a_conv1')(inputs)
    down0a = BatchNormalization()(down0a)
    down0a = LeakyReLU()(down0a)
    down0a = Conv2D(16, 3, padding='same', kernel_initializer='he_normal', name='down0a_conv2')(down0a)
    down0a = BatchNormalization()(down0a)
    down0a = LeakyReLU()(down0a)
    down0a_pool = MaxPooling2D(2, strides=2)(down0a)
    # 256

    down0 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal', name='down0_conv1')(down0a_pool)
    down0 = BatchNormalization()(down0)
    down0 = LeakyReLU()(down0)
    down0 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal', name='down0_conv2')(down0)
    down0 = BatchNormalization()(down0)
    down0 = LeakyReLU()(down0)
    down0_pool = MaxPooling2D(2, strides=2)(down0)
    # 128

    down1 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal', name='down1_conv1')(down0_pool)
    down1 = BatchNormalization()(down1)
    down1 = LeakyReLU()(down1)
    down1 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal', name='down1_conv2')(down1)
    down1 = BatchNormalization()(down1)
    down1 = LeakyReLU()(down1)
    down1_pool = MaxPooling2D(2, strides=2)(down1)
    # 64

    down2 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal', name='down2_conv1')(down1_pool)
    down2 = BatchNormalization()(down2)
    down2 = LeakyReLU()(down2)
    down2 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal', name='down2_conv2')(down2)
    down2 = BatchNormalization()(down2)
    down2 = LeakyReLU()(down2)
    down2_pool = MaxPooling2D(2, strides=2)(down2)
    # 32

    down3 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal', name='down3_conv1')(down2_pool)
    down3 = BatchNormalization()(down3)
    down3 = LeakyReLU()(down3)
    down3 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal', name='down3_conv2')(down3)
    down3 = BatchNormalization()(down3)
    down3 = LeakyReLU()(down3)
    down3_pool = MaxPooling2D(2, strides=2)(down3)
    # 16

    down4 = Conv2D(512, 3, padding='same', kernel_initializer='he_normal', name='down4_conv1')(down3_pool)
    down4 = BatchNormalization()(down4)
    down4 = LeakyReLU()(down4)
    down4 = Conv2D(512, 3, padding='same', kernel_initializer='he_normal', name='down4_conv2')(down4)
    down4 = BatchNormalization()(down4)
    down4 = LeakyReLU()(down4)
    down4_pool = MaxPooling2D(2, strides=2)(down4)
    # 8

    center = Conv2D(512, 3, padding='same', kernel_initializer='he_normal', name='center_conv1')(down4_pool)
    center = BatchNormalization()(center)
    center = LeakyReLU()(center)
    center = Conv2D(512, 3, padding='same', kernel_initializer='he_normal', name='center_conv2')(center)
    center = BatchNormalization()(center)
    center = LeakyReLU()(center)
    # center

    up4 = UpSampling2D(2)(center)
    up4 = concatenate([down4, up4], axis=3)
    up4 = Conv2D(512, 3, padding='same', kernel_initializer='he_normal', name='up4_conv1')(up4)
    up4 = BatchNormalization()(up4)
    up4 = LeakyReLU()(up4)
    up4 = Conv2D(512, 3, padding='same', kernel_initializer='he_normal', name='up4_conv2')(up4)
    up4 = BatchNormalization()(up4)
    up4 = LeakyReLU()(up4)
    up4 = Conv2D(512, 3, padding='same', kernel_initializer='he_normal', name='up4_conv3')(up4)
    up4 = BatchNormalization()(up4)
    up4 = LeakyReLU()(up4)
    # 16

    up3 = UpSampling2D(2)(up4)
    up3 = concatenate([down3, up3], axis=3)
    up3 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal', name='up3_conv1')(up3)
    up3 = BatchNormalization()(up3)
    up3 = LeakyReLU()(up3)
    up3 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal', name='up3_conv2')(up3)
    up3 = BatchNormalization()(up3)
    up3 = LeakyReLU()(up3)
    up3 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal', name='up3_conv3')(up3)
    up3 = BatchNormalization()(up3)
    up3 = LeakyReLU()(up3)
    # 32

    up2 = UpSampling2D(2)(up3)
    up2 = concatenate([down2, up2], axis=3)
    up2 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal', name='up2_conv1')(up2)
    up2 = BatchNormalization()(up2)
    up2 = LeakyReLU()(up2)
    up2 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal', name='up2_conv2')(up2)
    up2 = BatchNormalization()(up2)
    up2 = LeakyReLU()(up2)
    up2 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal', name='up2_conv3')(up2)
    up2 = BatchNormalization()(up2)
    up2 = LeakyReLU()(up2)
    # 64

    up1 = UpSampling2D(2)(up2)
    up1 = concatenate([down1, up1], axis=3)
    up1 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal', name='up1_conv1')(up1)
    up1 = BatchNormalization()(up1)
    up1 = LeakyReLU()(up1)
    up1 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal', name='up1_conv2')(up1)
    up1 = BatchNormalization()(up1)
    up1 = LeakyReLU()(up1)
    up1 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal', name='up1_conv3')(up1)
    up1 = BatchNormalization()(up1)
    up1 = LeakyReLU()(up1)
    # 128

    up0 = UpSampling2D(2)(up1)
    up0 = concatenate([down0, up0], axis=3)
    up0 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal', name='up0_conv1')(up0)
    up0 = BatchNormalization()(up0)
    up0 = LeakyReLU()(up0)
    up0 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal', name='up0_conv2')(up0)
    up0 = BatchNormalization()(up0)
    up0 = LeakyReLU()(up0)
    up0 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal', name='up0_conv3')(up0)
    up0 = BatchNormalization()(up0)
    up0 = LeakyReLU()(up0)
    # 256

    up0a = UpSampling2D(2)(up0)
    up0a = concatenate([down0a, up0a], axis=3)
    up0a = Conv2D(16, 3, padding='same', kernel_initializer='he_normal', name='up0a_conv1')(up0a)
    up0a = BatchNormalization()(up0a)
    up0a = LeakyReLU()(up0a)
    up0a = Conv2D(16, 3, padding='same', kernel_initializer='he_normal', name='up0a_conv2')(up0a)
    up0a = BatchNormalization()(up0a)
    up0a = LeakyReLU()(up0a)
    up0a = Conv2D(16, 3, padding='same', kernel_initializer='he_normal', name='up0a_conv3')(up0a)
    up0a = BatchNormalization()(up0a)
    up0a = LeakyReLU()(up0a)
    # 512

    classify = Conv2D(num_classes, 1, activation='sigmoid', name='output')(up0a)

    model = Model(inputs=inputs, outputs=classify)
    model.compile(optimizer=SGD(0.01, momentum=0.9, nesterov=True, decay=0.1), loss=bce_dice_loss, metrics=[dice_coef])
    model.summary()

    return model



def get_seq_unet_512(input_shape=(3, 512, 512, 3), num_classes=1):

    inputs = Input(shape=input_shape, name='input')
    # 512 

    down0a = TimeDistributed(Conv2D(16, 3, padding='same', kernel_initializer='he_normal', trainable=False), name='down0a_conv1')(inputs)
    down0a = TimeDistributed(BatchNormalization())(down0a)
    down0a = TimeDistributed(LeakyReLU())(down0a)
    down0a = TimeDistributed(Conv2D(16, 3, padding='same', kernel_initializer='he_normal', trainable=False), name='down0a_conv2')(down0a)
    down0a = TimeDistributed(BatchNormalization())(down0a)
    down0a = TimeDistributed(LeakyReLU())(down0a)
    down0a_pool = TimeDistributed(MaxPooling2D(2, strides=2))(down0a)
    # 256          
                   
    down0 = TimeDistributed(Conv2D (32, 3, padding='same', kernel_initializer='he_normal', trainable=False), name='down0_conv1')(down0a_pool)
    down0 = TimeDistributed(BatchNormalization())(down0)
    down0 = TimeDistributed(LeakyReLU())(down0)
    down0 = TimeDistributed(Conv2D(32, 3, padding='same', kernel_initializer='he_normal', trainable=False), name='down0_conv2')(down0)
    down0 = TimeDistributed(BatchNormalization())(down0)
    down0 = TimeDistributed(LeakyReLU())(down0)
    down0_pool = TimeDistributed(MaxPooling2D(2, strides=2))(down0)
    # 128

    down1 = TimeDistributed(Conv2D(64, 3, padding='same', kernel_initializer='he_normal', trainable=False), name='down1_conv1')(down0_pool)
    down1 = TimeDistributed(BatchNormalization())(down1)
    down1 = TimeDistributed(LeakyReLU())(down1)
    down1 = TimeDistributed(Conv2D(64, 3, padding='same', kernel_initializer='he_normal', trainable=False), name='down1_conv2')(down1)
    down1 = TimeDistributed(BatchNormalization())(down1)
    down1 = TimeDistributed(LeakyReLU())(down1)
    down1_pool = TimeDistributed(MaxPooling2D(2, strides=2))(down1)
    # 64

    down2 = TimeDistributed(Conv2D(128, 3, padding='same', kernel_initializer='he_normal', trainable=False), name='down2_conv1')(down1_pool)
    down2 = TimeDistributed(BatchNormalization())(down2)
    down2 = TimeDistributed(LeakyReLU())(down2)
    down2 = TimeDistributed(Conv2D(128, 3, padding='same', kernel_initializer='he_normal', trainable=False), name='down2_conv2')(down2)
    down2 = TimeDistributed(BatchNormalization())(down2)
    down2 = TimeDistributed(LeakyReLU())(down2)
    down2_pool = TimeDistributed(MaxPooling2D(2, strides=2))(down2)
    # 32

    down3 = TimeDistributed(Conv2D(256, 3, padding='same', kernel_initializer='he_normal', trainable=False), name='down3_conv1')(down2_pool)
    down3 = TimeDistributed(BatchNormalization())(down3)
    down3 = TimeDistributed(LeakyReLU())(down3)
    down3 = TimeDistributed(Conv2D(256, 3, padding='same', kernel_initializer='he_normal', trainable=False), name='down3_conv2')(down3)
    down3 = TimeDistributed(BatchNormalization())(down3)
    down3 = TimeDistributed(LeakyReLU())(down3)
    down3_pool = TimeDistributed(MaxPooling2D(2, strides=2))(down3)
    # 16

    down4 = TimeDistributed(Conv2D(512, 3, padding='same', kernel_initializer='he_normal', trainable=False), name='down4_conv1')(down3_pool)
    down4 = TimeDistributed(BatchNormalization())(down4)
    down4 = TimeDistributed(LeakyReLU())(down4)
    down4 = TimeDistributed(Conv2D(512, 3, padding='same', kernel_initializer='he_normal', trainable=False), name='down4_conv2')(down4)
    down4 = TimeDistributed(BatchNormalization())(down4)
    down4 = TimeDistributed(LeakyReLU())(down4)
    down4_pool = TimeDistributed(MaxPooling2D(2, strides=2))(down4)
    # 8

    center = TimeDistributed(Conv2D(512, 3, padding='same', kernel_initializer='he_normal', trainable=False), name='center_conv1')(down4_pool)
    center = TimeDistributed(BatchNormalization())(center)
    center = TimeDistributed(LeakyReLU())(center)
    center = TimeDistributed(Conv2D(512, 3, padding='same', kernel_initializer='he_normal', trainable=False), name='center_conv2')(center)
    center = TimeDistributed(BatchNormalization())(center)
    center = TimeDistributed(LeakyReLU())(center)
    # center

    up4 = TimeDistributed(UpSampling2D(2))(center)
    up4 = concatenate([down4, up4], axis=4)
    up4 = TimeDistributed(Conv2D(512, 3, padding='same', kernel_initializer='he_normal', trainable=False), name='up4_conv1')(up4)
    up4 = TimeDistributed(BatchNormalization())(up4)
    up4 = TimeDistributed(LeakyReLU())(up4)
    up4 = TimeDistributed(Conv2D(512, 3, padding='same', kernel_initializer='he_normal', trainable=False), name='up4_conv2')(up4)
    up4 = TimeDistributed(BatchNormalization())(up4)
    up4 = TimeDistributed(LeakyReLU())(up4)
    up4 = TimeDistributed(Conv2D(512, 3, padding='same', kernel_initializer='he_normal', trainable=False), name='up4_conv3')(up4)
    up4 = TimeDistributed(BatchNormalization())(up4)
    up4 = TimeDistributed(LeakyReLU())(up4)
    # 16

    up3 = TimeDistributed(UpSampling2D(2))(up4)
    up3 = concatenate([down3, up3], axis=4)
    up3 = TimeDistributed(Conv2D(256, 3, padding='same', kernel_initializer='he_normal', trainable=False), name='up3_conv1')(up3)
    up3 = TimeDistributed(BatchNormalization())(up3)
    up3 = TimeDistributed(LeakyReLU())(up3)
    up3 = TimeDistributed(Conv2D(256, 3, padding='same', kernel_initializer='he_normal', trainable=False), name='up3_conv2')(up3)
    up3 = TimeDistributed(BatchNormalization())(up3)
    up3 = TimeDistributed(LeakyReLU())(up3)
    up3 = TimeDistributed(Conv2D(256, 3, padding='same', kernel_initializer='he_normal', trainable=False), name='up3_conv3')(up3)
    up3 = TimeDistributed(BatchNormalization())(up3)
    up3 = TimeDistributed(LeakyReLU())(up3)
    # 32

    up2 = TimeDistributed(UpSampling2D(2))(up3)
    up2 = concatenate([down2, up2], axis=4)
    up2 = TimeDistributed(Conv2D(128, 3, padding='same', kernel_initializer='he_normal', trainable=False), name='up2_conv1')(up2)
    up2 = TimeDistributed(BatchNormalization())(up2)
    up2 = TimeDistributed(LeakyReLU())(up2)
    up2 = TimeDistributed(Conv2D(128, 3, padding='same', kernel_initializer='he_normal', trainable=False), name='up2_conv2')(up2)
    up2 = TimeDistributed(BatchNormalization())(up2)
    up2 = TimeDistributed(LeakyReLU())(up2)
    up2 = TimeDistributed(Conv2D(128, 3, padding='same', kernel_initializer='he_normal', trainable=False), name='up2_conv3')(up2)
    up2 = TimeDistributed(BatchNormalization())(up2)
    up2 = TimeDistributed(LeakyReLU())(up2)
    # 64

    up1 = TimeDistributed(UpSampling2D(2))(up2)
    up1 = concatenate([down1, up1], axis=4)
    up1 = TimeDistributed(Conv2D(64, 3, padding='same', kernel_initializer='he_normal', trainable=False), name='up1_conv1')(up1)
    up1 = TimeDistributed(BatchNormalization())(up1)
    up1 = TimeDistributed(LeakyReLU())(up1)
    up1 = TimeDistributed(Conv2D(64, 3, padding='same', kernel_initializer='he_normal', trainable=False), name='up1_conv2')(up1)
    up1 = TimeDistributed(BatchNormalization())(up1)
    up1 = TimeDistributed(LeakyReLU())(up1)
    up1 = TimeDistributed(Conv2D(64, 3, padding='same', kernel_initializer='he_normal', trainable=False), name='up1_conv3')(up1)
    up1 = TimeDistributed(BatchNormalization())(up1)
    up1 = TimeDistributed(LeakyReLU())(up1)
    # 128

    up0 = TimeDistributed(UpSampling2D(2))(up1)
    up0 = concatenate([down0, up0], axis=4)
    up0 = TimeDistributed(Conv2D(32, 3, padding='same', kernel_initializer='he_normal', trainable=False), name='up0_conv1')(up0)
    up0 = TimeDistributed(BatchNormalization())(up0)
    up0 = TimeDistributed(LeakyReLU())(up0)
    up0 = TimeDistributed(Conv2D(32, 3, padding='same', kernel_initializer='he_normal', trainable=False), name='up0_conv2')(up0)
    up0 = TimeDistributed(BatchNormalization())(up0)
    up0 = TimeDistributed(LeakyReLU())(up0)
    up0 = TimeDistributed(Conv2D(32, 3, padding='same', kernel_initializer='he_normal', trainable=False), name='up0_conv3')(up0)
    up0 = TimeDistributed(BatchNormalization())(up0)
    up0 = TimeDistributed(LeakyReLU())(up0)
    # 256

    up0a = TimeDistributed(UpSampling2D(2))(up0)
    up0a = concatenate([down0a, up0a], axis=4)
    up0a = TimeDistributed(Conv2D(16, 3, padding='same', kernel_initializer='he_normal', trainable=False), name='up0a_conv1')(up0a)
    up0a = TimeDistributed(BatchNormalization())(up0a)
    up0a = TimeDistributed(LeakyReLU())(up0a)
    up0a = TimeDistributed(Conv2D(16, 3, padding='same', kernel_initializer='he_normal', trainable=False), name='up0a_conv2')(up0a)
    up0a = TimeDistributed(BatchNormalization())(up0a)
    up0a = TimeDistributed(LeakyReLU())(up0a)
    up0a = TimeDistributed(Conv2D(16, 3, padding='same', kernel_initializer='he_normal', trainable=False), name='up0a_conv3')(up0a)
    up0a = TimeDistributed(BatchNormalization())(up0a)
    up0a = TimeDistributed(LeakyReLU())(up0a)
    # 512

    classify = TimeDistributed(Conv2D(num_classes, 1, activation='sigmoid', trainable=False), name='output')(up0a)

    #classify = Bidirectional(ConvLSTM2D(5, 3, padding='same', activation=None, kernel_initializer='he_normal', return_sequences=True))(classify)
    #classify = TimeDistributed(BatchNormalization())(classify)
    #classify = TimeDistributed(LeakyReLU())(classify)

    #classify = TimeDistributed(Conv2D(num_classes, 1, activation='sigmoid'))(up0a)

    model = Model(inputs=inputs, outputs=classify)
    model.compile(optimizer=SGD(0.01, momentum=0.9, nesterov=True), loss=bce_dice_loss, metrics=[dice_coef])
    model.summary()

    return model


def get_unet_1024(input_shape=(1024, 1024, 3), num_classes=1):
    inputs = Input(shape=input_shape, name='input')
    # 1024

    down0b = Conv2D(16, 3, padding='same', kernel_initializer='he_uniform', name='down0b_conv1')(inputs)
    down0b = BatchNormalization(name='down0b_norm1')(down0b)
    down0b = LeakyReLU()(down0b)
    down0b = Conv2D(16, 3, padding='same', kernel_initializer='he_uniform', name='down0b_conv2')(down0b)
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
    down4_pool = Dropout(0.25)(down4_pool)
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
    up0b = Conv2D(16, 3, padding='same', kernel_initializer='he_uniform', name='up0b_conv1')(up0b)
    up0b = BatchNormalization(name='up0b_norm1')(up0b)
    up0b = LeakyReLU()(up0b)
    up0b = Conv2D(16, 3, padding='same', kernel_initializer='he_uniform', name='up0b_conv2')(up0b)
    up0b = BatchNormalization(name='up0b_norm2')(up0b)
    up0b = LeakyReLU()(up0b)
    up0b = Conv2D(16, 3, padding='same', kernel_initializer='he_uniform', name='up0b_conv3')(up0b)
    up0b = BatchNormalization(name='up0b_norm3')(up0b)
    up0b = LeakyReLU()(up0b)
    # 1024

    classify = Conv2D(num_classes, 1, activation='sigmoid', name='output')(up0b)

    model = Model(inputs=inputs, outputs=classify)
    model.compile(optimizer=SGD(0.01, momentum=0.9, nesterov=True),
                  loss=bce_dice_loss, 
                  metrics=[dice_coef, false_positive, false_negative])
    model.summary()

    return model


def get_seq_unet_1024(input_shape=(5, 1024, 1024, 3), num_classes=1, unet_trainable=True, lstm_trainable=True):
    inputs = Input(shape=input_shape, name='input')
    # 1024

    down0b = TimeDistributed(Conv2D(8, 3, padding='same', kernel_initializer='he_normal', trainable=unet_trainable), name='down0b_conv1')(inputs)
    down0b = TimeDistributed(BatchNormalization(trainable=unet_trainable), name='down0b_norm1')(down0b)
    down0b = TimeDistributed(LeakyReLU())(down0b)
    down0b = TimeDistributed(Conv2D(8, 3, padding='same', kernel_initializer='he_normal', trainable=unet_trainable), name='down0b_conv2')(down0b)
    down0b = TimeDistributed(BatchNormalization(trainable=unet_trainable), name='down0b_norm2')(down0b)
    down0b = TimeDistributed(LeakyReLU())(down0b)
    down0b_pool = TimeDistributed(MaxPooling2D(2, strides=2))(down0b)
    # 512

    down0a = TimeDistributed(Conv2D(16, 3, padding='same', kernel_initializer='he_normal', trainable=unet_trainable), name='down0a_conv1')(down0b_pool)
    down0a = TimeDistributed(BatchNormalization(trainable=unet_trainable), name='down0a_norm1')(down0a)
    down0a = TimeDistributed(LeakyReLU())(down0a)
    down0a = TimeDistributed(Conv2D(16, 3, padding='same', kernel_initializer='he_normal', trainable=unet_trainable), name='down0a_conv2')(down0a)
    down0a = TimeDistributed(BatchNormalization(trainable=unet_trainable), name='down0a_norm2')(down0a)
    down0a = TimeDistributed(LeakyReLU())(down0a)
    down0a_pool = TimeDistributed(MaxPooling2D(2, strides=2))(down0a)
    # 256

    down0 = TimeDistributed(Conv2D(32, 3, padding='same', kernel_initializer='he_normal', trainable=unet_trainable), name='down0_conv1')(down0a_pool)
    down0 = TimeDistributed(BatchNormalization(trainable=unet_trainable), name='down0_norm1')(down0)
    down0 = TimeDistributed(LeakyReLU())(down0)
    down0 = TimeDistributed(Conv2D(32, 3, padding='same', kernel_initializer='he_normal', trainable=unet_trainable), name='down0_conv2')(down0)
    down0 = TimeDistributed(BatchNormalization(trainable=unet_trainable), name='down0_norm2')(down0)
    down0 = TimeDistributed(LeakyReLU())(down0)
    down0_pool = TimeDistributed(MaxPooling2D(2, strides=2))(down0)
    # 128

    down1 = TimeDistributed(Conv2D(64, 3, padding='same', kernel_initializer='he_normal', trainable=unet_trainable), name='down1_conv1')(down0_pool)
    down1 = TimeDistributed(BatchNormalization(trainable=unet_trainable), name='down1_norm1')(down1)
    down1 = TimeDistributed(LeakyReLU())(down1)
    down1 = TimeDistributed(Conv2D(64, 3, padding='same', kernel_initializer='he_normal', trainable=unet_trainable), name='down1_conv2')(down1)
    down1 = TimeDistributed(BatchNormalization(trainable=unet_trainable), name='down1_norm2')(down1)
    down1 = TimeDistributed(LeakyReLU())(down1)
    down1_pool = TimeDistributed(MaxPooling2D(2, strides=2))(down1)
    # 64

    down2 = TimeDistributed(Conv2D(128, 3, padding='same', kernel_initializer='he_normal', trainable=unet_trainable), name='down2_conv1')(down1_pool)
    down2 = TimeDistributed(BatchNormalization(trainable=unet_trainable), name='down2_norm1')(down2)
    down2 = TimeDistributed(LeakyReLU())(down2)
    down2 = TimeDistributed(Conv2D(128, 3, padding='same', kernel_initializer='he_normal', trainable=unet_trainable), name='down2_conv2')(down2)
    down2 = TimeDistributed(BatchNormalization(trainable=unet_trainable), name='down2_norm2')(down2)
    down2 = TimeDistributed(LeakyReLU())(down2)
    down2_pool = TimeDistributed(MaxPooling2D(2, strides=2))(down2)
    # 32

    down3 = TimeDistributed(Conv2D(256, 3, padding='same', kernel_initializer='he_normal', trainable=unet_trainable), name='down3_conv1')(down2_pool)
    down3 = TimeDistributed(BatchNormalization(trainable=unet_trainable), name='down3_norm1')(down3)
    down3 = TimeDistributed(LeakyReLU())(down3)
    down3 = TimeDistributed(Conv2D(256, 3, padding='same', kernel_initializer='he_normal', trainable=unet_trainable), name='down3_conv2')(down3)
    down3 = TimeDistributed(BatchNormalization(trainable=unet_trainable), name='down3_norm2')(down3)
    down3 = TimeDistributed(LeakyReLU())(down3)
    down3_pool = TimeDistributed(MaxPooling2D(2, strides=2))(down3)
    # 16

    down4 = TimeDistributed(Conv2D(512, 3, padding='same', kernel_initializer='he_normal', trainable=unet_trainable), name='down4_conv1')(down3_pool)
    down4 = TimeDistributed(BatchNormalization(trainable=unet_trainable), name='down4_norm1')(down4)
    down4 = TimeDistributed(LeakyReLU())(down4)
    down4 = TimeDistributed(Conv2D(512, 3, padding='same', kernel_initializer='he_normal', trainable=unet_trainable), name='down4_conv2')(down4)
    down4 = TimeDistributed(BatchNormalization(trainable=unet_trainable), name='down4_norm2')(down4)
    down4 = TimeDistributed(LeakyReLU())(down4)
    down4_pool = TimeDistributed(MaxPooling2D(2, strides=2))(down4)
    # 8

    center = TimeDistributed(Conv2D(1024, 3, padding='same', kernel_initializer='he_normal', trainable=unet_trainable), name='center_conv1')(down4_pool)
    center = TimeDistributed(BatchNormalization(trainable=unet_trainable), name='center_norm1')(center)
    center = TimeDistributed(LeakyReLU())(center)
    center = TimeDistributed(Conv2D(1024, 3, padding='same', kernel_initializer='he_normal', trainable=unet_trainable), name='center_conv2')(center)
    center = TimeDistributed(BatchNormalization(trainable=unet_trainable), name='center_norm2')(center)
    center = TimeDistributed(LeakyReLU())(center)
    # center

    up4 = TimeDistributed(UpSampling2D(2))(center)
    up4 = concatenate([down4, up4], axis=4)
    up4 = TimeDistributed(Conv2D(512, 3, padding='same', kernel_initializer='he_normal', trainable=unet_trainable), name='up4_conv1')(up4)
    up4 = TimeDistributed(BatchNormalization(trainable=unet_trainable), name='up4_norm1')(up4)
    up4 = TimeDistributed(LeakyReLU())(up4)
    up4 = TimeDistributed(Conv2D(512, 3, padding='same', kernel_initializer='he_normal', trainable=unet_trainable), name='up4_conv2')(up4)
    up4 = TimeDistributed(BatchNormalization(trainable=unet_trainable), name='up4_norm2')(up4)
    up4 = TimeDistributed(LeakyReLU())(up4)
    up4 = TimeDistributed(Conv2D(512, 3, padding='same', kernel_initializer='he_normal', trainable=unet_trainable), name='up4_conv3')(up4)
    up4 = TimeDistributed(BatchNormalization(trainable=unet_trainable), name='up4_norm3')(up4)
    up4 = TimeDistributed(LeakyReLU())(up4)
    # 16

    up3 = TimeDistributed(UpSampling2D(2))(up4)
    up3 = concatenate([down3, up3], axis=4)
    up3 = TimeDistributed(Conv2D(256, 3, padding='same', kernel_initializer='he_normal', trainable=unet_trainable), name='up3_conv1')(up3)
    up3 = TimeDistributed(BatchNormalization(trainable=unet_trainable), name='up3_norm1')(up3)
    up3 = TimeDistributed(LeakyReLU())(up3)
    up3 = TimeDistributed(Conv2D(256, 3, padding='same', kernel_initializer='he_normal', trainable=unet_trainable), name='up3_conv2')(up3)
    up3 = TimeDistributed(BatchNormalization(trainable=unet_trainable), name='up3_norm2')(up3)
    up3 = TimeDistributed(LeakyReLU())(up3)
    up3 = TimeDistributed(Conv2D(256, 3, padding='same', kernel_initializer='he_normal', trainable=unet_trainable), name='up3_conv3')(up3)
    up3 = TimeDistributed(BatchNormalization(trainable=unet_trainable), name='up3_norm3')(up3)
    up3 = TimeDistributed(LeakyReLU())(up3)
    # 32

    up2 = TimeDistributed(UpSampling2D(2))(up3)
    up2 = concatenate([down2, up2], axis=4)
    up2 = TimeDistributed(Conv2D(128, 3, padding='same', kernel_initializer='he_normal', trainable=unet_trainable), name='up2_conv1')(up2)
    up2 = TimeDistributed(BatchNormalization(trainable=unet_trainable), name='up2_norm1')(up2)
    up2 = TimeDistributed(LeakyReLU())(up2)
    up2 = TimeDistributed(Conv2D(128, 3, padding='same', kernel_initializer='he_normal', trainable=unet_trainable), name='up2_conv2')(up2)
    up2 = TimeDistributed(BatchNormalization(trainable=unet_trainable), name='up2_norm2')(up2)
    up2 = TimeDistributed(LeakyReLU())(up2)
    up2 = TimeDistributed(Conv2D(128, 3, padding='same', kernel_initializer='he_normal', trainable=unet_trainable), name='up2_conv3')(up2)
    up2 = TimeDistributed(BatchNormalization(trainable=unet_trainable), name='up2_norm3')(up2)
    up2 = TimeDistributed(LeakyReLU())(up2)
    # 64

    up1 = TimeDistributed(UpSampling2D(2))(up2)
    up1 = concatenate([down1, up1], axis=4)
    up1 = TimeDistributed(Conv2D(64, 3, padding='same', kernel_initializer='he_normal', trainable=unet_trainable), name='up1_conv1')(up1)
    up1 = TimeDistributed(BatchNormalization(trainable=unet_trainable), name='up1_norm1')(up1)
    up1 = TimeDistributed(LeakyReLU())(up1)
    up1 = TimeDistributed(Conv2D(64, 3, padding='same', kernel_initializer='he_normal', trainable=unet_trainable), name='up1_conv2')(up1)
    up1 = TimeDistributed(BatchNormalization(trainable=unet_trainable), name='up1_norm2')(up1)
    up1 = TimeDistributed(LeakyReLU())(up1)
    up1 = TimeDistributed(Conv2D(64, 3, padding='same', kernel_initializer='he_normal', trainable=unet_trainable), name='up1_conv3')(up1)
    up1 = TimeDistributed(BatchNormalization(trainable=unet_trainable), name='up1_norm3')(up1)
    up1 = TimeDistributed(LeakyReLU())(up1)
    # 128

    up0 = TimeDistributed(UpSampling2D(2))(up1)
    up0 = concatenate([down0, up0], axis=4)
    up0 = TimeDistributed(Conv2D(32, 3, padding='same', kernel_initializer='he_normal', trainable=unet_trainable), name='up0_conv1')(up0)
    up0 = TimeDistributed(BatchNormalization(trainable=unet_trainable), name='up0_norm1')(up0)
    up0 = TimeDistributed(LeakyReLU())(up0)
    up0 = TimeDistributed(Conv2D(32, 3, padding='same', kernel_initializer='he_normal', trainable=unet_trainable), name='up0_conv2')(up0)
    up0 = TimeDistributed(BatchNormalization(trainable=unet_trainable), name='up0_norm2')(up0)
    up0 = TimeDistributed(LeakyReLU())(up0)
    up0 = TimeDistributed(Conv2D(32, 3, padding='same', kernel_initializer='he_normal', trainable=unet_trainable), name='up0_conv3')(up0)
    up0 = TimeDistributed(BatchNormalization(trainable=unet_trainable), name='up0_norm3')(up0)
    up0 = TimeDistributed(LeakyReLU())(up0)
    # 256

    up0a = TimeDistributed(UpSampling2D(2))(up0)
    up0a = concatenate([down0a, up0a], axis=4)
    up0a = TimeDistributed(Conv2D(16, 3, padding='same', kernel_initializer='he_normal', trainable=unet_trainable), name='up0a_conv1')(up0a)
    up0a = TimeDistributed(BatchNormalization(trainable=unet_trainable), name='up0a_norm1')(up0a)
    up0a = TimeDistributed(LeakyReLU())(up0a)
    up0a = TimeDistributed(Conv2D(16, 3, padding='same', kernel_initializer='he_normal', trainable=unet_trainable), name='up0a_conv2')(up0a)
    up0a = TimeDistributed(BatchNormalization(trainable=unet_trainable), name='up0a_norm2')(up0a)
    up0a = TimeDistributed(LeakyReLU())(up0a)
    up0a = TimeDistributed(Conv2D(16, 3, padding='same', kernel_initializer='he_normal', trainable=unet_trainable), name='up0a_conv3')(up0a)
    up0a = TimeDistributed(BatchNormalization(trainable=unet_trainable), name='up0a_norm3')(up0a)
    up0a = TimeDistributed(LeakyReLU())(up0a)
    # 512

    up0b = TimeDistributed(UpSampling2D(2))(up0a)
    up0b = concatenate([down0b, up0b], axis=4)
    up0b = TimeDistributed(Conv2D(8, 3, padding='same', kernel_initializer='he_normal', trainable=unet_trainable), name='up0b_conv1')(up0b)
    up0b = TimeDistributed(BatchNormalization(trainable=unet_trainable), name='up0b_norm1')(up0b)
    up0b = TimeDistributed(LeakyReLU())(up0b)
    up0b = TimeDistributed(Conv2D(8, 3, padding='same', kernel_initializer='he_normal', trainable=unet_trainable), name='up0b_conv2')(up0b)
    up0b = TimeDistributed(BatchNormalization(trainable=unet_trainable), name='up0b_norm2')(up0b)
    up0b = TimeDistributed(LeakyReLU())(up0b)
    up0b = TimeDistributed(Conv2D(8, 3, padding='same', kernel_initializer='he_normal', trainable=unet_trainable), name='up0b_conv3')(up0b)
    up0b = TimeDistributed(BatchNormalization(trainable=unet_trainable), name='up0b_norm3')(up0b)
    up0b = TimeDistributed(LeakyReLU())(up0b)
    # 1024

    #classify = TimeDistributed(Conv2D(num_classes, 1, activation='sigmoid', trainable=unet_trainable), name='output')(up0b)

    classify = Bidirectional(ConvLSTM2D(4, 3, padding='same', activation=None, kernel_initializer='he_normal', return_sequences=True))(up0b)
    classify = TimeDistributed(BatchNormalization())(classify)
    classify = TimeDistributed(LeakyReLU())(classify)

    classify = TimeDistributed(Conv2D(num_classes, 1, activation='sigmoid'))(classify)

    model = Model(inputs=inputs, outputs=classify)
    model.compile(optimizer=SGD(0.001, momentum=0.9, nesterov=True),
                  loss=bce_dice_loss,
                  metrics=[dice_coef, false_positive, false_negative])
    model.summary()

    return model


def get_tiramisu(input_shape=(1024, 1024, 3), growth_rate=[2,4,4,16,16,16], depth_list=[2,4,6,8,10,10], dropout_p=0.5, wd=1e-4):
    inputs = Input(shape=input_shape)

    layer = Conv2D(48, 3, padding='same', kernel_initializer='he_uniform')(inputs)

    layer, skip_connections = down_path(layer, (4,5,7,10,12), [16]*5, [None]*5, dropout_p, wd=wd)
    layer, layer_list       = dense_block(layer, 15, 16, dropout_p, wd=wd)

    layer = up_path(layer_list, reversed(skip_connections), [12,10,7,5,4], [16]*5, [None]*5, dropout_p, wd=wd)

    classify = Conv2D(1, 1, activation='sigmoid', kernel_regularizer=l2(wd))(layer)

    model = Model(inputs=inputs, outputs=classify)
    model.compile(optimizer=RMSprop(lr=1e-3, decay=0.1),#SGD(0.01, momentum=0.9, nesterov=True),
                  loss=bce_dice_loss,
                  metrics=[dice_coef, false_positive, false_negative])
    model.summary(line_length=200)

    return model
"""
def relu(x): return Activation('relu')(x)
def dropout(x, p): return Dropout(p)(x) if p else x
def bn(x): return BatchNormalization(axis=-1)(x)
def relu_bn(x): return relu(bn(x))
def concat(xs): return merge(xs, mode='concat', concat_axis=-1)
def conv(x, nf, sz, wd, p, stride=1):
    x = Convolution2D(nf, sz, sz, init='he_uniform', border_mode='same',
                      subsample=(stride,stride), W_regularizer=l2(wd))(x)
    return dropout(x, p)

def conv_relu_bn(x, nf, sz=3, wd=0, p=0, stride=1):
    return conv(relu_bn(x), nf, sz, wd=wd, p=p, stride=stride)
def dense_block(n,x,growth_rate,p,wd):
    added = []
    for i in range(n):
        b = conv_relu_bn(x, growth_rate, p=p, wd=wd)
        x = concat([x, b])
        added.append(b)
    return x,added
def transition_dn(x, p, wd):
#     x = conv_relu_bn(x, x.get_shape().as_list()[-1], sz=1, p=p, wd=wd)
#     return MaxPooling2D(strides=(2, 2))(x)
    return conv_relu_bn(x, x.get_shape().as_list()[-1], sz=1, p=p, wd=wd, stride=2)
def down_path(x, nb_layers, growth_rate, p, wd):
    skips = []
    for i,n in enumerate(nb_layers):
        x,added = dense_block(n,x,growth_rate,p,wd)
        skips.append(x)
        x = transition_dn(x, p=p, wd=wd)
    return skips, added
def transition_up(added, wd=0):
    x = concat(added)
    _,r,c,ch = x.get_shape().as_list()
    return Deconvolution2D(ch, 3, 3, (None,r*2,c*2,ch), init='he_uniform',
               border_mode='same', subsample=(2,2), W_regularizer=l2(wd))(x)
#     x = UpSampling2D()(x)
#     return conv(x, ch, 2, wd, 0)
def up_path(added, skips, nb_layers, growth_rate, p, wd):
    for i,n in enumerate(nb_layers):
        x = transition_up(added, wd)
        x = concat([x,skips[i]])
        x,added = dense_block(n,x,growth_rate,p,wd)
    return x
def reverse(a): return list(reversed(a))


def create_tiramisu(nb_classes, img_input, nb_dense_block=6,
                    growth_rate=16, nb_filter=48, nb_layers_per_block=5, p=None, wd=0):
    if type(nb_layers_per_block) is list or type(nb_layers_per_block) is tuple:
        nb_layers = list(nb_layers_per_block)
    else:
        nb_layers = [nb_layers_per_block] * nb_dense_block

    x = conv(img_input, nb_filter, 3, wd, 0)
    skips, added = down_path(x, nb_layers, growth_rate, p, wd)
    x = up_path(added, reverse(skips[:-1]), reverse(nb_layers[:-1]), growth_rate, p, wd)

    x = conv(x, nb_classes, 1, wd, 0)

    return Activation('sigmoid')(x)

def get_tiramisu2():
    input_shape = (256, 256, 3)
    img_input = Input(shape=input_shape)
    x = create_tiramisu(1, img_input, nb_layers_per_block=[4, 5, 7, 10, 12, 15], p=0.2, wd=1e-4)
    model = Model(img_input, x)
    model.compile(optimizer=RMSprop(lr=1e-3, decay=1.-0.99995),  # SGD(0.01, momentum=0.9, nesterov=True),
                  loss=bce_dice_loss,
                  metrics=[dice_coef, false_positive, false_negative])
    model.summary(line_length=200)

    return model
"""
