import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, UpSampling2D, Dropout, ReLU


model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))


def unet(input_shape, output_channels):
    
    """
    - All convolutions are 4×4 spatialfilters applied with stride 2.
    
    """
    # ----------------------------------------------------------------
    # SETTINGS
    
    # Convolutional layer
    strides = 2
    kernel_size = (4, 4) 
    padding = 'same' # Todo: check padding, should be 1
    # The data_loader returns inputs with shape (batch, height, width, channels)
    df = "channels_last"
    
    # Upsampling
    ups_size = 2
    
    # Batch Normalisation
    # TODO: Instance Normalisation: batchnorm use test stats at test time
    # TODO: pytorch implementation has learnable params, how to implement in keras?
    bn_axis = -1    # because data_loader returns channels last
    bn_mmtm = 0.9   # equivalent to pytorch defaults used by author (0.1 in pytorch -> 0.9 in keras/tf)
    bn_eps  = 1e-5  # match pytorch defaults
    
    # ReLU
    slope = 0.2
    
    # Skip connections
    merge_mode = 'concat'
    
    # Dropout
    dropout = 0.5
    
    # ----------------------------------------------------------------
    # ENCODER
    
    # - Let Ck denote a Convolution-BatchNorm-ReLU layerwith k filter
    # - Convolutions downsample by a factor of 2
    # - All ReLUs are leaky
    # Architecture is then:
    # C64-C128-C256-C512-C512-C512-C512-C512
    
    # TODO: check padding is correct
    # TODO: check downsampling/upsampling behaviour is as expected
    
    # layer 1 - C64
    # Batch-Norm is not applied to the first C64 layer
    x1 = Conv2D(64, kernel_size=kernel_size, strides=strides, padding=padding, data_format=df input_shape=input_shape)
    x1 = LeakyReLU(alpha=slope)(x1)
    
    # layer 2 - C128
    x2 = Conv2D(128, kernel_size=kernel_size, strides=strides, padding=padding, data_format=df)(x1)
    x2 = BatchNormalization(axis=bn_axis, momentum=bn_mmtm, epsilon=bn_eps)(x2)
    x2 = LeakyReLU(alpha=slope)(x2)
    
    # layer 3 - C256
    x3 = Conv2D(256, kernel_size=kernel_size, strides=strides, padding=padding, data_format=df)(x2)
    x3 = BatchNormalization(axis=bn_axis, momentum=bn_mmtm, epsilon=bn_eps)(x3)
    x3 = LeakyReLU(alpha=slope)(x3)
    
    # layer 4 - C512
    x4 = Conv2D(512, kernel_size=kernel_size, strides=strides, padding=padding, data_format=df)(x3)
    x4 = BatchNormalization(axis=bn_axis, momentum=bn_mmtm, epsilon=bn_eps)(x4)
    x4 = LeakyReLU(alpha=slope)(x4)
    
    # layer 5 - C512
    x5 = Conv2D(512, kernel_size=kernel_size, strides=strides, padding=padding, data_format=df)(x4)
    x5 = BatchNormalization(axis=bn_axis, momentum=bn_mmtm, epsilon=bn_eps)(x5)
    x5 = LeakyReLU(alpha=slope)(x5)
    
    # layer 6 - C512
    x6 = Conv2D(512, kernel_size=kernel_size, strides=strides, padding=padding, data_format=df)(x5)
    x6 = BatchNormalization(axis=bn_axis, momentum=bn_mmtm, epsilon=bn_eps)(x6)
    x6 = LeakyReLU(alpha=slope)(x6)
    
    # layer 7 - C512
    x7 = Conv2D(512, kernel_size=kernel_size, strides=strides, padding=padding, data_format=df)(x6)
    x7 = BatchNormalization(axis=bn_axis, momentum=bn_mmtm, epsilon=bn_eps)(x7)
    x7 = LeakyReLU(alpha=slope)(x7)
    
    # layer 8 - C512    
    x8 = Conv2D(512, kernel_size=kernel_size, strides=strides, padding=padding, data_format=df)(x7)
    x8 = BatchNormalization(axis=bn_axis, momentum=bn_mmtm, epsilon=bn_eps)(x8)
    x8 = LeakyReLU(alpha=slope)(x8)   
    
    # ----------------------------------------------------------------
    # DECODER
    
    # - Ck denotes a Convolution-BatchNorm-ReLU layerwith k filters:
    # - CDk denotes a a Convolution-BatchNorm-Dropout-ReLU layer 
    #   with a dropout rate of 50%
    # - Convolutions upsample by a factor of 2,
    # - All ReLUs are not leaky
    # Architecture is then:
    # CD512-CD1024-CD1024-C1024-C1024-C512-C256-C128
    
    # layer 9 - CD512
    x9 = UpSampling2D(size=ups_size, data_format=df)(x8) 
    x9 = Conv2D(512, kernel_size=kernel_size, strides=strides, padding=padding, data_format=df)(x9)
    x9 = BatchNormalization(axis=bn_axis, momentum=bn_mmtm, epsilon=bn_eps)(x9)
    x9 = Dropout(p=dropout)(x9)
    x9 = ReLU()(x9)
    
    # TODO: Where should skip connections be concatenated?
    # Should updampling layers have stride 2? pix2pix implementation suggests yes
    
    # layer 10 - CD1024
    # layer 11 - CD1024
    # layer 12 - C1024
    # layer 13 - C1024
    # layer 14 - C512
    # layer 15 - C256
    # layer 16 - C128
    
    
    
    
    # After the last layer in the decoder, a convolution is applied 
    # to map to the number of output channels (3 in general, except 
    # in colorization, where it is 2), followed by a Tanh function. 