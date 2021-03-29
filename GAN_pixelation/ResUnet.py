from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import *

def ResUnet(init_ch):
    """
    Encoder: Residual blocks + MaxPooling
    Skip connection: add
    Decoder: Residual blocks + ConvTranspose
    :param input_shape:
    :param init_ch: число фильтров в первой слое
    :return:
    """

    def ReSBlocK(x, filedepth):
        """
        Residual block with 2 conv
        :param x: input layer
        :param filedepth: #filters
        :return:
        """
        conv1 = Conv2D(filters=filedepth, kernel_size=[1, 1], strides=1, padding='same')(x)
        conv2 = Conv2D(filters=filedepth, kernel_size=[3, 3], strides=1, padding='same')(conv1)
        conv2 = Activation('relu')(BatchNormalization()(conv2))
        conv3 = Conv2D(filters=filedepth, kernel_size=[3, 3], strides=1, padding='same')(conv2)
        conv3 = BatchNormalization()(conv3)
        conv3 = Activation('relu')(Add()([conv1, conv3]))
        return conv3

    input_ = Input(shape=(None, None, 3))
    conv1 = ReSBlocK(input_, init_ch)  #######
    down2 = Conv2D(init_ch, kernel_size=2, strides=2)(conv1)

    conv2 = ReSBlocK(down2, 2 * init_ch)  ######
    down3 = Conv2D(2*init_ch, kernel_size=2, strides=2)(conv2)

    conv3 = ReSBlocK(down3, 4 * init_ch)  #####
    down4 = Conv2D(4*init_ch, kernel_size=2, strides=2)(conv3)

    conv4 = ReSBlocK(down4, 8 * init_ch)  ####
    down5 = Conv2D(8*init_ch, kernel_size=2, strides=2)(conv4)


    conv6 = ReSBlocK(down5, 12 * init_ch)  ##


    up7 = Concatenate(axis=-1)([UpSampling2D(size=2, interpolation='nearest')(conv6), conv4])  ####
    conv8 = ReSBlocK(up7, 8 * init_ch)

    up8 = Concatenate(axis=-1)([UpSampling2D(size=2, interpolation='nearest')(conv8), conv3])  #####
    conv9 = ReSBlocK(up8, 4 * init_ch)

    up9 = Concatenate(axis=-1)([UpSampling2D(size=2, interpolation='nearest')(conv9), conv2])  ######
    conv10 = ReSBlocK(up9, 2 * init_ch)

    up10 = Concatenate(axis=-1)([UpSampling2D(size=2, interpolation='nearest')(conv10), conv1])  #######
    conv11 = ReSBlocK(up10, init_ch)

    conv_ref = Conv2D(init_ch, (3, 3), strides=1, padding='same')(conv11)
    conv_ref = Activation('relu')(BatchNormalization()(conv_ref))
    conv_ref = Conv2D(init_ch, (3, 3), strides=1, padding='same')(conv_ref)
    conv_ref = Activation('relu')(BatchNormalization()(conv_ref))
    conv_out = Conv2D(1, (1, 1), activation='linear', strides=1, padding='same')(conv_ref)
    model = Model(inputs=input_, outputs=conv_out)
    return model

if __name__ == '__main__':
    model = ResUnet(4)
    model.compile(optimizer=Adam(lr=1e-4), loss='mean_squared_error')
    model.summary()
    exit(0)