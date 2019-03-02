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
#from model.generator_ternaus import ternausNet16 as unet

class Pix2Pix():
    def __init__(self):
        # Input shape
        self.img_rows = 256
        self.img_cols = 256
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        # Data loader
        # ---------------------------------------------------------
        self.dataset_name = 'facades'
        self.sample_dir = f'images/{self.dataset_name}'
        self.experiment_title = 'pix2pix_patchgan2'
        self.data_loader = DataLoader(dataset_name=self.dataset_name,
                                      img_res=(self.img_rows, self.img_cols))


        # Build Discriminator
        # ---------------------------------------------------------
        optimizer_d = Adam(lr=0.0002, beta_1=0.5)
        self.discriminator = build_discriminator(input_size=(256, 256, 3))
        self.discriminator.compile(loss='mse', optimizer=optimizer_d, metrics=['accuracy'])

        # Build Generator
        # ---------------------------------------------------------        
        self.generator = build_generator(input_size=(256, 256, 3), output_channels=3)
        
        # Build combined GAN
        # ---------------------------------------------------------
        
        # Input conditioning image
        input_ = Input(shape=self.img_shape)
        # Generate fake image
        output = self.generator(input_)

        # For the combined only train the generator
        self.discriminator.trainable = False
        self.discriminator.compile(loss='mse', optimizer=optimizer_d, metrics=['accuracy'])

        # Is z fake or real given condition the x?
        is_real = self.discriminator([output, input_])

        optimizer_g = Adam(lr=0.0002, beta_1=0.5)
        self.combined = Model(inputs=[input_], outputs=[output, is_real])
        self.combined.compile(loss=['mae', 'mse'], loss_weights=[100, 1], optimizer=optimizer_g)


    def train(self, epochs, batch_size=1, sample_interval=50):

        start_time = datetime.datetime.now()

        # Adversarial loss ground truths (patchgan70 outputs 32x32x1)
        discriminator_output_sz = (32, 32, 1)
        real = np.ones((batch_size, ) + discriminator_output_sz)
        fake = np.zeros((batch_size, ) + discriminator_output_sz)

        for epoch in range(epochs):
            for batch_i, (targets, inputs) in enumerate(self.data_loader.load_batch(batch_size)):


                #  Train Discriminator
                # ----------------------------------------------

                # Condition on x and generate a translated version
                outputs, _ = self.combined.predict(inputs)

                # Train the discriminators (original images = real / generated = Fake)
                d_loss_real = self.discriminator.train_on_batch([targets, inputs], real)
                d_loss_fake = self.discriminator.train_on_batch([outputs, inputs], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                
                #  Train Generator
                # ----------------------------------------------

                # Train the generators in GAN setting
                g_loss = self.combined.train_on_batch([inputs], [targets, real])

                elapsed_time = datetime.datetime.now() - start_time
                # Plot the progress
                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s" % (epoch, epochs,
                                                                        batch_i, self.data_loader.n_batches,
                                                                        d_loss[0], 100*d_loss[1],
                                                                        g_loss[0],
                                                                        elapsed_time))

                # If at save interval => save generated image samples
                if batch_i % sample_interval == 0:
                    self.sample_images(self.sample_dir, epoch, batch_i)

    def sample_images(self, sample_dir, epoch, batch_i):
        os.makedirs(sample_dir, exist_ok=True)
        r, c = 3, 3

        imgs_A, imgs_B = self.data_loader.load_data(batch_size=3, is_testing=True)
        fake_A = self.generator.predict(imgs_B)

        gen_imgs = np.concatenate([imgs_B, fake_A, imgs_A])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Condition', 'Generated', 'Original']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt])
                axs[i, j].set_title(titles[i])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig(f'{sample_dir}/{str(epoch).zfill(4)}_{batch_i}_{self.experiment_title}.png')
        plt.close()


if __name__ == '__main__':
    gan = Pix2Pix()
    gan.train(epochs=200, batch_size=1, sample_interval=200)