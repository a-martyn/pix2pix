import numpy as np
import os
import datetime
import argparse

import tensorflow as tf
from tensorflow.python.keras.engine.network import Network
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from model.data_loader import dataLoader
from model.generator_pix2pix import unet_pix2pix as build_generator
from model.discriminator_patchgan import bceWithLogitsLoss
from model.discriminator_patchgan import patchgan70 as build_discriminator
from evaluate import evaluate
from utils import Metrics




# GAN SETUP
# ---------------------------------------------------------
"""
NOTE:
We need to ensure that the discriminator has no trainable weights 
within the GAN model, but is trainable in isolation.

This is tricky to achieve in Keras because and error is thrown
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


# Options
# ---------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--title', type=str, required=True, help='Title used to name results of this experiment')
parser.add_argument('--norm_type', type=str, default='instance', help='Type of normalisation used in generator model [instance, batch]')
parser.add_argument('--d_loss', type=str, default='BCE', help='Type of loss used for discriminator model [BCE, MSE]')
parser.add_argument('--mbstd', action='store_true', help='Bool: Use minibatch standard deviation to increase variance')
args = parser.parse_args()

experiment_title = args.title
norm_type = args.norm_type

if args.d_loss == 'BCE':
    print('USING: BCE Loss')
    d_loss_fn = bceWithLogitsLoss
elif args.d_loss == 'MSE':
    print('USING: MSE Loss')
    d_loss_fn = 'mean_squared_error'
else:
    raise NotImplementedError(f'Supported d_loss arg: [SCE, MSE]')

minibatch_std=args.mbstd
if minibatch_std: print('\nUSING: minibatch_stddev_layer')

# Parameters
# ---------------------------------------------------------
# GPUs available
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

d_acc_min = 1.0
input_sz = (256, 256, 3)
discriminator_output_sz = (32, 32, 1)
epochs=200
batch_size=4
sample_interval=200

L1_loss_weight = 100
GAN_loss_weight = 1

dataset_name = 'facades'
train_metrics_pth = f'results/{dataset_name}/{experiment_title}_train.csv'
val_metrics_pth = f'results/{dataset_name}/{experiment_title}_val.csv'
sample_dir = f'results/{dataset_name}/images'
train_pth = 'data/facades_processed/train'
val_pth = 'data/facades_processed/val'
n_samples = 400

d_lr = 0.00002
d_beta1 = 0.5
g_lr = 0.0002
g_beta1 = 0.5

# Data loader
# ---------------------------------------------------------
train_generator = ImageDataGenerator(
    rescale=1./255,
    zoom_range=[0.895, 1.0],
    horizontal_flip=True,
    fill_mode='constant',
    data_format='channels_last',
    validation_split=0.0
)
train_loader = dataLoader(train_pth, train_generator, 
                          batch_sz=batch_size, img_sz=input_sz[:2])

val_generator = ImageDataGenerator(
    rescale=1./255,
    fill_mode='constant',
    data_format='channels_last',
    validation_split=0.0
)
val_loader = dataLoader(val_pth, val_generator, 
                        batch_sz=1, img_sz=input_sz[:2])




# Build and compile Discriminator
# ---------------------------------------------------------
inputs_dsc, outputs_dsc = build_discriminator(input_size=input_sz, 
                                              minibatch_std=minibatch_std)
discriminator = Model(inputs_dsc, outputs_dsc, name='discriminator')

optimizer_d = Adam(lr=d_lr, beta_1=d_beta1)
discriminator.compile(loss=d_loss_fn, optimizer=optimizer_d, metrics=['accuracy'])
discriminator.summary()

# Debug 1/3: Record trainable weights in discriminator
ntrainable_dsc = len(discriminator.trainable_weights)


# Build frozen Discriminator without learnable params
# ---------------------------------------------------------
frozen_discriminator = Network(inputs_dsc, outputs_dsc, 
                               name='frozen_discriminator')
frozen_discriminator.trainable = False


# Build Generator
# ---------------------------------------------------------        
input_gen, output_gen = build_generator(norm_type=norm_type, 
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

optimizer_g = Adam(lr=g_lr, beta_1=g_beta1)
gan.compile(loss=['mean_absolute_error', d_loss_fn], 
            loss_weights=[L1_loss_weight, GAN_loss_weight], 
            optimizer=optimizer_g)

# Debug 3/3: assert that...
# The discriminator has trainable weights
assert(len(discriminator._collected_trainable_weights) == ntrainable_dsc)
# Only the generators weights are trainable in the GAN model
assert(len(gan._collected_trainable_weights) == ntrainable_gen)


print(f'discriminator.metrics_names: {discriminator.metrics_names}')
print(f'gan.metrics_names: {gan.metrics_names}')

# Train
# --------------------------------------------------------
train_metrics = Metrics(train_metrics_pth)
val_metrics = Metrics(val_metrics_pth)
start_time = datetime.datetime.now()

for epoch in range(epochs):
    for batch in range(n_samples):
        # ganhack2: modified loss function/label flip real => 0
        # ganhack: label smoothing
        real = np.random.random_sample((batch_size, ) + discriminator_output_sz) * 0.3  # label smoothing: real => 0.0 - 0.3
        fake = np.ones((batch_size, ) + discriminator_output_sz)   # fake => 1
        
        inputs, targets = next(train_loader)
        outputs, _ = gan.predict(inputs)
        
#         # Eval discriminator
#         d_loss_real = discriminator.evaluate([targets, inputs], real, verbose=0)
#         d_loss_fake = discriminator.evaluate([outputs, inputs], fake, verbose=0)
#         d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)       
        
#         d_trained = False
#         if d_loss[1] <= d_acc_min:
        
        #  Train Discriminator
        # ----------------------------------------------
        d_trained = True
        # Train the discriminators (original images = real / generated = Fake)
        d_loss_real, d_acc_real = discriminator.train_on_batch([targets, inputs], real)
        d_loss_fake, d_acc_fake = discriminator.train_on_batch([outputs, inputs], fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        
        #  Train Generator
        # ----------------------------------------------

        # Train the generators in GAN setting
        g_loss = gan.train_on_batch([inputs], [targets, real])

        elapsed_time = datetime.datetime.now() - start_time
        # Plot the progress
        
        train_metrics.add({
            'epoch': epoch,
            'batch': batch,
            'D_trained': d_trained,
            'D_loss_real': d_loss,
            'D_loss_real': d_loss_real,
            'D_loss_fake': d_loss_fake,
            'D_acc_fake': d_acc_fake,
            'G_total_loss': g_loss[0],
            'G_L1_loss': g_loss[1],
            'G_Disc_loss': g_loss[2],
            'time': elapsed_time
        })


        # If at save interval => save generated image samples
        if batch % sample_interval == 0:
            train_metrics.to_csv()
            real_labels = np.zeros((1, ) + discriminator_output_sz) # no label smoothing at test time
            evaluate(gan, discriminator, val_loader, real_labels, sample_dir, epoch, batch, experiment_title, val_metrics)
