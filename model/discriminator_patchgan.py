import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *


def minibatch_stddev_layer(x, group_size=4):
    """
    Adapted from: https://github.com/tkarras/progressive_growing_of_gans/blob/master/networks.py
    """
    group_size = tf.minimum(group_size, tf.shape(x)[0])     # Minibatch must be divisible by (or smaller than) group_size.
    s = x.shape                                             # [NHWC]  Input shape.
    y = tf.reshape(x, [group_size, -1, s[1], s[2], s[3]])   # [GMHWC] Split minibatch into M groups of size G.
    y = tf.cast(y, tf.float32)                              # [GMHWC] Cast to FP32.
    y -= tf.reduce_mean(y, axis=0, keepdims=True)           # [GMHWC] Subtract mean over group.
    y = tf.reduce_mean(tf.square(y), axis=0)                # [MHWC]  Calc variance over group.
    y = tf.sqrt(y + 1e-8)                                   # [MHWC]  Calc stddev over group.
    y = tf.reduce_mean(y, axis=[1,2,3], keepdims=True)      # [M111]  Take average over fmaps and pixels.
    y = tf.cast(y, x.dtype)                                 # [M111]  Cast back to original data type.
    y = tf.tile(y, [group_size, s[1], s[2], 1])             # [NHW1]  Replicate over group and pixels.
    return tf.concat([x, y], axis=-1)                       # [NHWC]  Append as new fmap.


def conv_layer(x, out_channels, kernel_size, strides=2, batch_norm=True, 
               init='random_normal'):
    
    conv_kwargs = dict(
        padding='same',
        kernel_initializer=init,
        bias_initializer=init,
        data_format='channels_last'  # (batch, height, width, channels)
    )
    bn_kwargs = dict(
        axis=-1,       # because data_loader returns channels last
        momentum=0.1,  # equivalent to pytorch defaults used by author 
        epsilon=1e-5,   # match pytorch/torch defaults
        gamma_initializer=tf.random_normal_initializer(1.0, 0.02),
        trainable=True
    )
    
    x = Conv2D(out_channels, kernel_size, strides=strides, **conv_kwargs)(x)
    if batch_norm: x = BatchNormalization(**bn_kwargs)(x)
    x = LeakyReLU(alpha=0.2)(x)
    return x
    
    
def patchgan70(input_size=(256, 256, 3), init_gain=0.02, minibatch_std=False):
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
    init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=init_gain)

    # MODEL
    # ---------------------------------------------
    img_A = Input(input_size)
    img_B = Input(input_size)
    
    x = Concatenate(axis=-1)([img_A, img_B])                       # (256, 256, real_channels+fake_channels)
    x = conv_layer(x,  64, 4, strides=2, batch_norm=False, init=init)         # (128, 128,  64) TRF=4
    x = conv_layer(x, 128, 4, strides=2, batch_norm=True, init=init)          # ( 64,  64, 128) TRF=10
    #if minibatch_std: x = Lambda(minibatch_stddev_layer)(x)
    x = conv_layer(x, 256, 4, strides=2, batch_norm=True, init=init)          # ( 32,  32, 256) TRF=22
    x = conv_layer(x, 512, 4, strides=1, batch_norm=True, init=init)          # ( 32,  32, 512) TRF=46 
    
    op = Conv2D(1, 4, strides=1, padding='same', name='D_logits', 
                kernel_initializer=init, bias_initializer=init)(x)           # ( 32,   32,  1) TRF=70
    # no activation because using BCE with logits
    
    inputs=[img_A, img_B]
    outputs=[op]
    return inputs, outputs


def bceWithLogitsLoss(y_true, y_logits):
    """
    Equivalent to PyTorch's nn.BCEWithLogitsLoss
    """
    return K.binary_crossentropy(
        y_true,
        y_logits,
        from_logits=True
    )