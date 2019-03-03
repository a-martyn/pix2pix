from __future__ import print_function, division
import scipy

import tensorflow as tf
from tensorflow.python.keras.engine.network import Network
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt
import sys
from data_loader import DataLoader
#from model.data_loader import DataGenerator
import numpy as np
import pandas as pd
import os
from skimage.transform import resize

from model.generator_pix2pix import unet_pix2pix as build_generator
from model.discriminator_patchgan import patchgan70 as build_discriminator



def sample_images(gan, dataloader, sample_dir, epoch, batch_i, experiment_title):
    """
    Sample 3 images from generator and save to sample_dir
    """
    # Run a forward pass
    # -----------------------
    targets, inputs = data_loader.load_data(batch_size=3, is_testing=True)
    outputs, is_real = gan.predict(inputs)

    # Save images to disk
    # -----------------------
    os.makedirs(sample_dir, exist_ok=True)
    r, c = 4, 3
    
    # Create heatmap from patchgan discriminator output
    print(f'Patch Size: {is_real[0].shape}')
    patches = [resize(x, (256, 256, 1), order=0, preserve_range=True) for x in is_real]
    patches = np.asarray(patches)
    patches = np.concatenate([patches, patches, patches], axis=-1) # convert to rgb
    
    imgs = np.concatenate([inputs, outputs, patches, targets])

    # Rescale images 0 - 1
    imgs = 0.5 * imgs + 0.5

    titles = ['Condition', 'Generated', 'Patches', 'Original']
    fig, axs = plt.subplots(r, c, figsize=(10, 10))
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(imgs[cnt])
            axs[i,j].axis('off')
            if i == 2:
                axs[i,j].set_title(f'{titles[i]}: {np.mean(is_real[j]): .2f}')
            else:
                axs[i, j].set_title(titles[i])
            cnt += 1
    fig.savefig(f'{sample_dir}/{str(epoch).zfill(4)}_{batch_i}_{experiment_title}.png')
    plt.close()
    return


def train(discriminator, gan, dataloader, sample_dir, experiment_title, epochs=200, batch_size=1, sample_interval=50):

    start_time = datetime.datetime.now()

    # Adversarial loss ground truths (patchgan70 outputs 32x32x1)
    discriminator_output_sz = (32, 32, 1)
    real = np.ones((batch_size, ) + discriminator_output_sz)
    fake = np.zeros((batch_size, ) + discriminator_output_sz)

    for batch_i, (targets, inputs) in enumerate(data_loader.load_batch(batch_size)):
        
        #  Train Discriminator
        # ----------------------------------------------

        # Condition on x and generate a translated version
        outputs, _ = gan.predict(inputs)

        # Train the discriminators (original images = real / generated = Fake)
        d_loss_real = discriminator.train_on_batch([targets, inputs], real)
        d_loss_fake = discriminator.train_on_batch([outputs, inputs], fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        
        #  Train Generator
        # ----------------------------------------------

        # Train the generators in GAN setting
        g_loss = gan.train_on_batch([inputs], [targets, real])

        elapsed_time = datetime.datetime.now() - start_time
        # Plot the progress
        
        print((
            f'{epoch}/{epochs} | '
            f'{batch_i}/{data_loader.n_batches} | '
            f'DISCRIMINATOR loss: {d_loss[0] :.5f} acc: {d_loss[1] :.5f} | '
            f'GENERATOR loss: {g_loss[0] :.5f} unet_loss: {g_loss[1] :.5f} patchgan_loss: {g_loss[2] :.5f}  | '
            f'time: {elapsed_time}'
        ))


        # If at save interval => save generated image samples
        if batch_i % sample_interval == 0:
            sample_images(gan, dataloader, sample_dir, epoch, batch_i, experiment_title)


# GAN SETUP
# ---------------------------------------------------------
"""
NOTE:
We need to ensure that the discriminator has no trainable weights 
within the GAN model, but is trainable in isolation.

This is tricky to achive in Keras because and error is thrown
when the discriminator is set .trainable=False without compile
being called. To achieve this we follow these steps:

1. Build and compile the discriminator with learnble weights
2. Build a 'frozen' discriminator that has no learnable weights
3. Build a generator with learnable weights
4. Compile frozen discriminator and generator into GAN where only
   the generator learns.

Based on suggestion given here:
https://github.com/keras-team/keras/issues/8585#issuecomment-412728017
"""

# Parameters
# ---------------------------------------------------------
norm_type = 'batch'
input_sz = (256, 256, 3)
epochs=200
batch_size=1
sample_interval=200

L1_loss_weight = 100
GAN_loss_weight = 1

dataset_name = 'facades'
sample_dir = f'images/{dataset_name}'
experiment_title = 'baseline'


# Data loader
# ---------------------------------------------------------
data_loader = DataLoader(dataset_name=dataset_name, img_res=input_sz[:2])


# Build and compile Discriminator
# ---------------------------------------------------------
inputs_dsc, outputs_dsc = build_discriminator(input_size=input_sz)
discriminator = Model(inputs_dsc, outputs_dsc, name='discriminator')

optimizer_d = Adam(lr=0.0002, beta_1=0.5)
discriminator.compile(loss='binary_crossentropy', optimizer=optimizer_d, metrics=['accuracy'])
discriminator.summary()

# Debug 1/3: Record trainable weights in discriminator
ntrainable_dsc = len(discriminator.trainable_weights)


# Build frozen Discriminator without learnable params
# ---------------------------------------------------------
frozen_discriminator = Network(inputs_dsc, outputs_dsc, name='frozen_discriminator')
frozen_discriminator.trainable = False


# Build Generator
# ---------------------------------------------------------        
input_gen, output_gen = build_generator(norm_type='batch', 
                                        input_size=input_sz, 
                                        output_channels=input_sz[-1])
generator = Model(input_gen, output_gen, name='generator')
generator.summary()

# Debug  2/3: Record trainable weights in generator
ntrainable_gen = len(generator.trainable_weights)

# Build GAN and compile
# ---------------------------------------------------------  
is_real = frozen_discriminator([output_gen, input_gen])
gan = Model(input_gen, [output_gen, is_real], name='gan')
gan.summary()

optimizer_g = Adam(lr=0.0002, beta_1=0.5)
gan.compile(loss=['mae', 'binary_crossentropy'], 
            loss_weights=[L1_loss_weight, GAN_loss_weight], 
            optimizer=optimizer_g)

# Debug 3/3: assert that...
# The discriminator has trainable weights
assert(len(discriminator._collected_trainable_weights) == ntrainable_dsc)
# Only the generators weights are trainable in the GAN model
assert(len(gan._collected_trainable_weights) == ntrainable_gen)


# Train
# ---------------------------------------------------------
for epoch in range(epochs):
    train(discriminator, gan, data_loader, sample_dir, experiment_title, 
          epochs=epochs, batch_size=batch_size, sample_interval=sample_interval)


