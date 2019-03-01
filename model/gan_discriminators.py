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


def patchgan70(input_size=(256, 256, 3)):
    """
    PatchGAN with a 70x70 receptive field. This is used throughout paper
    except where a specific receptive field size is stated.

    # Notes: expects input to be fake+real concatenated channel-wise

    Questions
    - should input size be 512px?
    - if no, when authors say receptive field 70x70 do they mean 34x34?
    - authors use padding=1 throughout. What padding does keras 'same' 
      result in here?
    """
    # SETTINGS
    # ---------------------------------------------
    conv_kwargs = dict(
        kernel_size=4,
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
    img_A = Input(input_size)
    img_B = Input(input_size)
    concat = Concatenate(axis=-1)([img_A, img_B])
    # (256, 256, real_channels+fake_channels)
    x = Conv2D(64, strides=2, **conv_kwargs)(concat)
    x = LeakyReLU(alpha=leak_slope)(x)
    # (128, 128, 64)
    x = Conv2D(128, strides=2, **conv_kwargs)(x)
    x = BatchNormalization(**bn_kwargs)(x)
    x = LeakyReLU(alpha=leak_slope)(x)
    # (64, 64, 128)
    x = Conv2D(256, strides=2, **conv_kwargs)(x)
    x = BatchNormalization(**bn_kwargs)(x)
    x = LeakyReLU(alpha=leak_slope)(x)
    # (32, 32, 256)
    x = Conv2D(512, strides=1, **conv_kwargs)(x)
    x = BatchNormalization(**bn_kwargs)(x)
    x = LeakyReLU(alpha=leak_slope)(x)
    # (32, 32, 512)
    x = Conv2D(1, strides=1, **conv_kwargs)(x)
    op = Activation('sigmoid')(x)
    # (32, 32, 1)
    model = Model(inputs=[img_A, img_B], outputs=[op])
    return model
