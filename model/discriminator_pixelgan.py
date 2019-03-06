import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *


def pixelgan(input_size=(256, 256, 3)):
    """
    1x1 receptive field PixelGAN.

    Classifies each pixel individually.
    """

    # SETTINGS
    # ---------------------------------------------
    conv_kwargs = dict(
        kernel_size=1,
        strides=1,
        padding='same',
        kernel_initializer='he_normal',
        data_format='channels_last'  # (batch, height, width, channels)
    )
    bn_kwargs = dict(
        axis=-1,       # because data_loader returns channels last
        momentum=0.9,  # equivalent to pytorch defaults used by author
        epsilon=1e-5   # match pytorch defaults
    )

    leak_slope = 0.2
    
    # MODEL
    # ---------------------------------------------
    inputs = Input(input_size)
    # (256, 256, input_channels)
    x = Conv2D(64, **conv_kwargs)(inputs)
    x = LeakyReLU(alpha=leak_slope)(x)
    # (256, 256, 64)
    x = Conv2D(128, **conv_kwargs)(x)
    x = BatchNormalization(**bn_kwargs)(x)
    x = LeakyReLU(alpha=leak_slope)(x)
    # (256, 256, 128)
    x = Conv2D(1, **conv_kwargs)(x)
    op = Activation('sigmoid')(x)
    # (256, 256, 1)
    model = Model(inputs=[inputs], outputs=[op])
    return model


