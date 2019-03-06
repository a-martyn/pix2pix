import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *


def conv_layer(x, out_channels, kernel_size, strides=2, batch_norm=True):
    
    conv_kwargs = dict(
        padding='same',
        kernel_initializer='he_normal',
        data_format='channels_last'  # (batch, height, width, channels)
    )
    bn_kwargs = dict(
        axis=-1,       # because data_loader returns channels last
        momentum=0.9,  # equivalent to pytorch defaults used by author 
        epsilon=1e-5   # match pytorch/torch defaults
    )
    
    x = Conv2D(out_channels, kernel_size, strides=strides, **conv_kwargs)(x)
    if batch_norm: x = BatchNormalization(**bn_kwargs)(x)
    x = LeakyReLU(alpha=0.2)(x)
    return x
    
    
def patchgan70(input_size=(256, 256, 3)):
    """
    PatchGAN with a 70x70 receptive field. This is used throughout paper
    except where a specific receptive field size is stated. 
    
    Verified against authors' PyTorch and Torch implementations:
    - https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    - https://github.com/phillipi/pix2pix/blob/master/models.lua
    
    Questions
    - authors use padding=1 throughout. What padding does keras 'same' 
      result in here?
      
    Known discrepancies:
    - authors use batchnorm -> relu, as described in the original
      batchnorm paper. 
    """

    leak_slope = 0.2

    # MODEL
    # ---------------------------------------------
    img_A = Input(input_size)
    img_B = Input(input_size)
    
    x = Concatenate(axis=-1)([img_A, img_B])                       # (256, 256, real_channels+fake_channels)
    x = conv_layer(x,  64, 4, strides=2, batch_norm=False)         # (128, 128,  64) TRF=4
    x = conv_layer(x, 128, 4, strides=2, batch_norm=True)          # ( 64,  64, 128) TRF=10
    x = conv_layer(x, 256, 4, strides=2, batch_norm=True)          # ( 32,  32, 256) TRF=22
    x = conv_layer(x, 512, 4, strides=1, batch_norm=True)          # ( 32,  32, 512) TRF=46 
    
    op = Conv2D(1, 4, strides=1, padding='same')(x)                # ( 32,   32,  1) TRF=70
    # no activation because using BCE with logits
    #op = Activation('sigmoid')(op)
    
    inputs=[img_A, img_B]
    outputs=[op]
    return inputs, outputs

