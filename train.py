import numpy as np
import os
import datetime
import argparse

import tensorflow as tf
tf.enable_eager_execution()

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

import tensorflow.keras.backend as K
from tensorflow.python.keras.engine.network import Network
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from model.data_loader import dataLoader
from model.generator_pix2pix import unet_pix2pix as build_generator
from model.discriminator_patchgan import patchgan70 as build_discriminator
from evaluate import gen_checkpoint
from utils.metrics import Metrics
from utils.html import build_results_page



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


# Losses
# ----------------------------------

def bce_loss(y_true, y_pred):
    """ 
    Binary cross entropy loss.
    - y_true: array of floats between 0 and 1
    - y_preds: sigmoid activations output from model
    """
    EPS = 1e-12
    x = -tf.reduce_mean((y_true * tf.log(y_pred + EPS)) + ((1-y_true) * tf.log(1-y_pred + EPS)))
    return x


def l1_loss(y_true, y_pred):
    """ L1 Loss with mean reduction per PyTorch default """
    # abs(targets - outputs) => 0
    return tf.reduce_mean(tf.abs(y_true - y_pred))


def d_loss_fn(model, x, y_true):
    """ """
    EPS = 1e-12
    y_pred = model([x[0], x[1]])
    loss_disc = bce_loss(y_true, y_pred)
    return loss_disc

def g_loss_fn(model, x, y_true, lambda_L1):
    """  """
    # abs(targets - outputs) => 0
    EPS = 1e-12
    g_pred, d_pred = model(x)
    loss_L1 = l1_loss(y_true[0], g_pred) * lambda_L1
    loss_gan = bce_loss(y_true[1], d_pred)
    loss_total = loss_gan + loss_L1
    return loss_total, loss_L1, loss_gan


# Learning rate
# ----------------------------------
def gen_lr_schedule(lr_init, decay_start, decay_end, steps_per_epoch):
    """ """
    def get_lr(current_epoch, final_epoch, lr_init):
        return lr_init - ((lr_init / final_epoch) * current_epoch)
    
    decay_interval = decay_end - decay_start + 1
    epochs = np.arange(1, decay_interval, dtype=np.int32)
    boundaries = list(steps_per_epoch * (epochs + decay_start))[:-1]
    values = [get_lr(e, decay_interval, lr_init) for e in epochs]
    return boundaries, values


# Options
# ---------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--title', type=str, required=True, help='Title used to name results of this experiment')
parser.add_argument('--norm_type', type=str, default='instance', help='Type of normalisation used in generator model [instance, batch]')
parser.add_argument('--mbstd', action='store_true', help='Bool: Use minibatch standard deviation to increase variance')

parser.add_argument('--batch_size', default=1, type=int)
args = parser.parse_args()

experiment_title = args.title
norm_type = args.norm_type

minibatch_std=args.mbstd
if minibatch_std: print('\nUSING: minibatch_stddev_layer')

# Parameters
# ---------------------------------------------------------
# GPUs available
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

d_acc_min = 1.0
input_sz = (256, 256, 3)
discriminator_output_sz = (30, 30, 1)
epochs=200
batch_size=args.batch_size
sample_interval=400

lambda_L1 = 100.0   # weight applied to L1 loss in gan

dataset_name = 'facades'
train_metrics_pth = f'results/{dataset_name}/{experiment_title}_train.csv'
val_metrics_pth = f'results/{dataset_name}/{experiment_title}_val.csv'
sample_dir = f'results/{dataset_name}/images'
train_pth = 'data/facades_processed/train'
# val_pth = 'data/facades_processed/val'
checkpoints_pth = f'results/{dataset_name}/checkpoints/images'
metric_keys = ['G_total', 'G_L1', 'G_GAN', 'D_loss'] 
metrics_plt_pth = f'results/{dataset_name}/checkpoints/metrics.png'
n_samples = 400

lr = 0.0002
lr_beta1 = 0.5
lr_decay_start = 100
lr_decay_end = 200
seed = 9678 #np.random.randint(1, 10000)


# Data loader
# ---------------------------------------------------------

train_generator = ImageDataGenerator(
    rescale=1./255,
    zoom_range=[0.8, 1.0],
    horizontal_flip=True,
    fill_mode='constant',
    data_format='channels_last',
    validation_split=0.0
)
train_loader = dataLoader(train_pth, train_generator, batch_sz=batch_size, 
                          shuffle=True, img_sz=input_sz[:2], seed=seed)

d_loader = dataLoader(train_pth, train_generator, batch_sz=batch_size, 
                      shuffle=True, img_sz=input_sz[:2], seed=seed+21)


check_generator = ImageDataGenerator(
    rescale=1./255,
    fill_mode='constant',
    data_format='channels_last',
    validation_split=0.0
)

check_loader = dataLoader(checkpoints_pth, check_generator, batch_sz=1, 
                          shuffle=False, img_sz=(256, 256), seed=seed)


# Learning rate annealing
# ---------------------------------------------------------
global_step = tf.Variable(0, trainable=False)

# Linear reduction in learning rate to zero after 100th epoch
boundaries, values = gen_lr_schedule(lr, lr_decay_start, lr_decay_end, n_samples)
# this is a function because we're in eager execution mode
lr_fn = tf.train.piecewise_constant(global_step, boundaries, values)

# Optimizers
# ---------------------------------------------------------

optimizer_d = tf.train.AdamOptimizer(learning_rate=lr_fn, beta1=lr_beta1, beta2=0.999, epsilon=1e-08)
optimizer_g = tf.train.AdamOptimizer(learning_rate=lr_fn, beta1=lr_beta1, beta2=0.999, epsilon=1e-08)


# Build discriminator
# ---------------------------------------------------------
d_input, d_output = build_discriminator(input_size=input_sz, 
                                          minibatch_std=minibatch_std)
discriminator = Model(d_input, d_output, name='discriminator')
discriminator.summary()


# Build Generator
# ---------------------------------------------------------        
g_input, g_output = build_generator(norm_type=norm_type, 
                                    input_size=input_sz, 
                                    output_channels=input_sz[-1])
generator = Model(g_input, g_output, name='generator')
generator.summary()


# Build GAN
# ---------------------------------------------------------  
g_output_gan = generator(g_input)
d_output_gan = discriminator([g_output_gan, g_input])
gan = Model(g_input, [g_output_gan, d_output_gan], name='gan')
gan.summary()


# Train
# --------------------------------------------------------

train_metrics = Metrics(train_metrics_pth)
start_time = datetime.datetime.now()

real = np.ones((batch_size, ) + discriminator_output_sz)   # real => 1
fake = np.zeros((batch_size, ) + discriminator_output_sz)  # fake => 0

for epoch in range(epochs):
    gen_checkpoint(gan, check_loader, epoch, checkpoints_pth)
    build_results_page(epoch)
    for batch in range(n_samples):                

        #  Train Discriminator
        # ----------------------------------------------
        inputs, targets = next(d_loader)
        
        outputs, _ = gan.predict(inputs)  
        
        #  Custom discriminator training
        # ----------------------------------------------
        with tf.GradientTape() as tape:
            d_loss_fake = d_loss_fn(discriminator, [outputs, inputs], fake)
            d_loss_real = d_loss_fn(discriminator, [targets, inputs], real)
            d_loss_total = 0.5 * (d_loss_fake + d_loss_real)

        grads = tape.gradient(d_loss_total, discriminator.trainable_variables)
        optimizer_d.apply_gradients(zip(grads, discriminator.trainable_variables))
               
        
        
        # Train Generator
        # ----------------------------------------------
        # Train the generators in GAN setting
        #g_loss = gan.train_on_batch([inputs], [targets, real])
 
        inputs, targets = next(train_loader)
        
        with tf.GradientTape() as tape:
            g_loss_total, g_loss_L1, g_loss_gan = g_loss_fn(gan, inputs, [targets, real], lambda_L1)

        grads = tape.gradient(g_loss_total, generator.trainable_variables)
        optimizer_g.apply_gradients(zip(grads, generator.trainable_variables), global_step=global_step)
        
        

        # Plot the progress
        if (batch+1) % 100 == 0:
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
    train_metrics.plot(metric_keys, metrics_plt_pth)
    

# Save models
gan.save(f'pretrained/{experiment_title}_gan.h5')
discriminator.save(f'pretrained/{experiment_title}_discriminator.h5')
