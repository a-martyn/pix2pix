import numpy as np
import os
import datetime
import argparse

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.keras.engine.network import Network
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from model.data_loader import dataLoader
from model.generator_pix2pix import unet_pix2pix as build_generator
from model.discriminator_patchgan import patchgan70 as build_discriminator
from model.losses import bce_loss, l1_loss, d_loss_fn, g_loss_fn
from model.learning_rate import gen_lr_schedule
from evaluate import gen_checkpoint
from utils.metrics import Metrics, print_setup
from utils.html import build_results_page

tf.enable_eager_execution()



# Command Line Interface
# ---------------------------------------------------------
parser = argparse.ArgumentParser()
# Paths
parser.add_argument('--experiment_title', type=str, required=True, help='Title used to name results of this experiment')
parser.add_argument('--train_pth', default='data/facades_processed/train', type=str, help='training dataset directory')
parser.add_argument('--checkpoints_pth', default='data/facades_processed/checkpoints', type=str, help='checkpoints dataset directory')
parser.add_argument('--pretrained_pth', default='pretrained', type=str, help='trained models saved to this directory')
parser.add_argument('--results_pth', default='results', type=str, help='training results saved to this directory')
# Dataset parameters
parser.add_argument('--n_samples', default=400, type=int, help='the number of observations in training set')
parser.add_argument('--input_size', default=256, type=int, help='width and height dimension of input array')
parser.add_argument('--input_channels', default=3, type=int, help='channels dimension of input array')
# Training parameters
parser.add_argument('--epochs', default=200, type=int, help='the total number of epochs to train for')
parser.add_argument('--sample_interval', default=100, type=int, help='record metrics every n training steps')
parser.add_argument('--batch_size', default=1, type=int, help='number of observations per batch')
parser.add_argument('--lr', default=0.0002, type=float, help='learning rate for both discriminator and generator')
parser.add_argument('--lr_beta1', default=0.5, type=float, help='Adam optimizer beta1 value')
parser.add_argument('--lr_decay_start', default=100, type=int, help='epoch on which to begin learning rate decay')
parser.add_argument('--lr_decay_end', default=200, type=int, help='epoch on which to end learning rate decay')
parser.add_argument('--lambda_L1', default=100.0, type=float, help='weight applied to L1 loss in gan')
parser.add_argument('--seed', default=9678, type=int, help='random seed for data loaders')


args = parser.parse_args()

experiment_title = args.experiment_title
train_pth        = args.train_pth
checkpoints_pth  = args.checkpoints_pth
pretrained_pth   = args.pretrained_pth
results_pth      = args.results_pth

epochs           = args.epochs              
n_samples        = args.n_samples         
sample_interval  = args.sample_interval 

input_sz         = (args.input_size, args.input_size, args.input_channels)
batch_size       = args.batch_size
lr               = args.lr
lr_beta1         = args.lr_beta1
lr_decay_start   = args.lr_decay_start
lr_decay_end     = args.lr_decay_end
lambda_L1        = args.lambda_L1
seed             = args.seed


# Parameters
# ---------------------------------------------------------
metrics_csv_pth = f'{results_pth}/logs/{experiment_title}_train.csv'
html_filepath = f'{results_pth}/index.html'
metrics_plt_filepath = f'{results_pth}/metrics.png'
# This determines which metrics are plotted in training results webpage (max 6)
metric_keys = ['G_total', 'G_L1', 'G_GAN', 'D_loss'] 

discriminator_output_sz = (30, 30, 1)   # dimensions of discriminator output
checkpoint_dir_labels = {
    'input': 'input', 
    'gen_pytorch': "authors' pytorch", 
    'gen_tf': 'this implementation', 
    'target': 'target', 
    'patch_tf': 'patchgan'
}

# Setup
# ---------------------------------------------------------

# Set number of GPUs available on this device
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
# make dir for pretrained model if it does not exist
os.makedirs(pretrained_pth, exist_ok=True)


# Data Augmentation
# ---------------------------------------------------------

train_generator = ImageDataGenerator(
    rescale=1./255,
    zoom_range=[0.8, 1.0],   # roughly equivalent to authors' enlarge and crop
    horizontal_flip=True,
    fill_mode='constant',
    data_format='channels_last',
    validation_split=0.0
)

check_generator = ImageDataGenerator(
    rescale=1./255,
    fill_mode='constant',
    data_format='channels_last',
    validation_split=0.0
)

# Data Loaders
# ---------------------------------------------------------

# At each step, generator and discriminator train on a different image.
# Not part of the original paper, but I just wanted to reasure myself 
# that disconnection here does not impeded performance.
g_loader = dataLoader(train_pth, train_generator, batch_sz=batch_size, 
                          shuffle=True, img_sz=input_sz[:2], seed=seed)

d_loader = dataLoader(train_pth, train_generator, batch_sz=batch_size, 
                      shuffle=True, img_sz=input_sz[:2], seed=seed+21)

# Load checkpoint images for training visualisation in
# results/facades/checkpoints/index.html
check_loader = dataLoader(checkpoints_pth, check_generator, batch_sz=1, 
                          shuffle=False, img_sz=(256, 256), seed=seed)


# Learning rate annealing
# ---------------------------------------------------------
global_step = tf.Variable(0, trainable=False)

# Linear reduction in learning rate to zero after 100th epoch
boundaries, values = gen_lr_schedule(lr, lr_decay_start, lr_decay_end, n_samples)
# lr_fn is a function because we're in eager execution mode
lr_fn = tf.train.piecewise_constant(global_step, boundaries, values)

# Optimizers
# ---------------------------------------------------------
optimizer_d = tf.train.AdamOptimizer(learning_rate=lr_fn, beta1=lr_beta1, 
                                     beta2=0.999, epsilon=1e-08)
optimizer_g = tf.train.AdamOptimizer(learning_rate=lr_fn, beta1=lr_beta1, 
                                     beta2=0.999, epsilon=1e-08)


# Build discriminator
# ---------------------------------------------------------
d_input, d_output = build_discriminator(input_size=input_sz)
discriminator = Model(d_input, d_output, name='discriminator')

# Build Generator
# ---------------------------------------------------------        
g_input, g_output = build_generator(input_size=input_sz, 
                                    output_channels=input_sz[-1])
generator = Model(g_input, g_output, name='generator')

# Build GAN
# --------------------------------------------------------- 
# The generator and discriminator are both in keras' trainable=True mode
# which means that batchnorm and dropout will be applied as intended.
# We will update only the generators weights during training of this model.
g_output_gan = generator(g_input)
d_output_gan = discriminator([g_output_gan, g_input])
gan = Model(g_input, [g_output_gan, d_output_gan], name='gan')


# Display setup details
# --------------------------------------------------------- 
print_setup(tf.__version__, 
            tf.executing_eagerly(), 
            args, 
            discriminator.count_params(), 
            generator.count_params(),
            gan.count_params())


# Training loop
# --------------------------------------------------------

# Initialize metrics
train_metrics = Metrics(metrics_csv_pth)
start_time = datetime.datetime.now()

# PatchGan discriminator labels, arrays of ones or zeroes
real = np.ones((batch_size, ) + discriminator_output_sz)   # real => 1
fake = np.zeros((batch_size, ) + discriminator_output_sz)  # fake => 0

for epoch in range(epochs):
    # update training results webpage
    gen_checkpoint(gan, check_loader, epoch, checkpoints_pth)
    build_results_page(epoch, checkpoints_pth, checkpoint_dir_labels, 
                       metrics_plt_filepath, html_filepath)
    
    for batch in range(n_samples):                

        #  Train Discriminator
        # ----------------------------------------------
        # Two forward passes for fake and real examples before a single 
        # backwards pass to update discriminators' weights
        inputs, targets = next(d_loader) 
        # Use gan to generate fake       
        outputs, _ = gan.predict(inputs)  
        
        with tf.GradientTape() as tape:
            d_loss_fake = d_loss_fn(discriminator, [outputs, inputs], fake)
            d_loss_real = d_loss_fn(discriminator, [targets, inputs], real)
            d_loss_total = 0.5 * (d_loss_fake + d_loss_real)

        grads = tape.gradient(d_loss_total, discriminator.trainable_variables)
        optimizer_d.apply_gradients(zip(grads, discriminator.trainable_variables))
               
        # Train Generator
        # ----------------------------------------------
        # Forward pass through GAN updating only the generator's weights on
        # the backward pass
        inputs, targets = next(g_loader)
        
        with tf.GradientTape() as tape:
            g_loss_total, g_loss_L1, g_loss_gan = g_loss_fn(gan, inputs, [targets, real], lambda_L1)

        grads = tape.gradient(g_loss_total, generator.trainable_variables)
        optimizer_g.apply_gradients(zip(grads, generator.trainable_variables), 
                                    global_step=global_step)
        
        # Record progress
        # ----------------------------------------------
        # Cache training metrics and print to terminal
        if (batch+1) % sample_interval == 0:
            elapsed_time = datetime.datetime.now() - start_time
            train_metrics.add({
                'epoch': epoch,
                'iters': batch+1,
                'G_lr': optimizer_g._lr_t.numpy(),
                'D_lr': optimizer_d._lr_t.numpy(),
                'G_L1': g_loss_L1.numpy(),
                'G_GAN': g_loss_gan.numpy(),
                'G_total': g_loss_total.numpy(),
                'D_loss': d_loss_total.numpy(),
                'time': elapsed_time,
                'random_seed': seed
        })

    train_metrics.to_csv()
    # Update plot displayed on training webpage
    train_metrics.plot(metric_keys, metrics_plt_filepath)
    
# Build final results page
# --------------------------------------------------------
gen_checkpoint(gan, check_loader, epochs, checkpoints_pth)
build_results_page(epochs, checkpoints_pth, checkpoint_dir_labels, 
                   metrics_plt_filepath, html_filepath)

# Save models
# --------------------------------------------------------
gan.save(f'{pretrained_pth}/{experiment_title}_gan.h5')
discriminator.save(f'{pretrained_pth}/{experiment_title}_discriminator.h5')
