from __future__ import print_function, division
import scipy

from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from keras_contrib.layers.normalization import InstanceNormalization
import datetime
import matplotlib.pyplot as plt
import sys
from data_loader import DataLoader
#from model.data_loader import DataGenerator
import numpy as np
import pandas as pd
import os

from model.generator_pix2pix import unet_pix2pix as build_generator
from model.discriminator_patchgan import patchgan70 as build_discriminator



def sample_images(gan, dataloader, sample_dir, epoch, batch_i, experiment_title):
    """
    Sample 3 images from generator and save to sample_dir
    """
    
    os.makedirs(sample_dir, exist_ok=True)
    r, c = 3, 3

    targets, inputs = data_loader.load_data(batch_size=3, is_testing=True)
    outputs, _ = gan.predict(inputs)

    imgs = np.concatenate([inputs, outputs, targets])

    # Rescale images 0 - 1
    imgs = 0.5 * imgs + 0.5

    titles = ['Condition', 'Generated', 'Original']
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(imgs[cnt])
            axs[i, j].set_title(titles[i])
            axs[i,j].axis('off')
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




# Parameters
# ---------------------------------------------------------
input_sz = (256, 256, 3)
epochs=200
batch_size=1
sample_interval=200

dataset_name = 'facades'
sample_dir = f'images/{dataset_name}'
experiment_title = 'pix2pix_patchgan3'


# Data loader
# ---------------------------------------------------------
data_loader = DataLoader(dataset_name=dataset_name, img_res=input_sz[:2])


# Build Discriminator
# ---------------------------------------------------------
optimizer_d = Adam(lr=0.0002, beta_1=0.5)
discriminator = build_discriminator(input_size=input_sz)
discriminator.compile(loss='mse', optimizer=optimizer_d, metrics=['accuracy'])

# Build Generator
# ---------------------------------------------------------        
generator = build_generator(input_size=input_sz, output_channels=input_sz[-1])


# Build combined GAN
# ---------------------------------------------------------
        
# Input conditioning image
input_ = Input(shape=input_sz)
# Generate fake image
output = generator(input_)

# For the gan only train the generator
discriminator.trainable = False
discriminator.compile(loss='mse', optimizer=optimizer_d, metrics=['accuracy'])

# Is output fake or real given condition the input_?
is_real = discriminator([output, input_])

optimizer_g = Adam(lr=0.0002, beta_1=0.5)
gan = Model(inputs=[input_], outputs=[output, is_real])
gan.compile(loss=['mae', 'mse'], loss_weights=[100, 1], optimizer=optimizer_g)


# Train
# ---------------------------------------------------------

for epoch in range(epochs):
    train(discriminator, gan, data_loader, sample_dir, experiment_title, 
          epochs=epochs, batch_size=batch_size, sample_interval=sample_interval)