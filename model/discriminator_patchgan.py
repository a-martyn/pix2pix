import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *


def conv_layer(x, out_channels, kernel_size, strides=2, batch_norm=True, 
               init='random_normal', use_bias=True):
    
    conv_kwargs = dict(
        use_bias=use_bias,
        padding='valid',
        kernel_initializer=init,
        bias_initializer=init,
        data_format='channels_last'  # (batch, height, width, channels)
    )
    
    bn_kwargs = dict(
        axis=-1,                   # because data_loader returns channels last
        momentum=0.9,              # 0.1 in pytorch -> 0.9 in keras/tf
        epsilon=1e-5,              # PyTorch default
        beta_initializer='zeros',
        gamma_initializer=tf.initializers.random_uniform(0.0, 1.0),  
        center=True,               # equivalent to affine=True
        scale=True,                # equivalent to affine=True
        trainable=True,
    )
    
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(out_channels, kernel_size, strides=strides, **conv_kwargs)(x)
    if batch_norm: x = BatchNormalization(**bn_kwargs)(x)
    x = LeakyReLU(alpha=0.2)(x)
    return x
    
    
def patchgan70(input_size=(256, 256, 3), init_gain=0.02):
    """
    PatchGAN with a 70x70 receptive field. This is used throughout paper
    except where a specific receptive field size is stated. 
    
    Cross referenced against authors' PyTorch and Torch implementations:
    - https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/ \
      models/networks.py
    - https://github.com/phillipi/pix2pix/blob/master/models.lua
    """

    leak_slope = 0.2
    init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=init_gain)

    # MODEL
    # ---------------------------------------------
    img_A = Input(input_size)
    img_B = Input(input_size)
    
    x = Concatenate(axis=-1)([img_A, img_B])                                           # (256, 256, real_channels+fake_channels)
    x = conv_layer(x,  64, 4, strides=2, batch_norm=False, init=init, use_bias=True)   # (128, 128,  64) TRF=4
    x = conv_layer(x, 128, 4, strides=2, batch_norm=True, init=init, use_bias=False)   # ( 64,  64, 128) TRF=10
    x = conv_layer(x, 256, 4, strides=2, batch_norm=True, init=init, use_bias=False)   # ( 32,  32, 256) TRF=22
    x = conv_layer(x, 512, 4, strides=1, batch_norm=True, init=init, use_bias=False)   # ( 31,  31, 512) TRF=46 
    
    op = ZeroPadding2D(padding=(1, 1))(x)
    op = Conv2D(1, 4, strides=1, padding='valid', name='D_logits', 
                kernel_initializer=init, bias_initializer=init, use_bias=True)(op)     # ( 30,   30,  1) TRF=70
    op = Activation('sigmoid', name='D_activations')(op)
    
    inputs=[img_A, img_B]
    outputs=[op]
    return inputs, outputs
