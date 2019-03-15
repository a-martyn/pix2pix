import numpy as np
import os
import datetime
import argparse

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.keras.engine.network import Network
from tensorflow.keras.layers import Input
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



# Learning rate
# ----------------------------------

class LrScheduler():
    """
    keep the same learning rate for the first <niter> epochs 
    and linearly decay the rate to zero over the next <niter_decay> epochs
    
    
             | ________________
             |                 \
             |                  \
          lr |                   \
             |                    \
             |--------------------------
                       epochs

    - init_lr: the initial learning rate
    - decay_start: number of epochs before learning rate reduction begins
    - decay_end: the epoch on which learning rate reaches zero
    
    """
    
    def __init__(self, init_lr: float, decay_start: int, decay_end: int):
        # linear so stepsize is constant
        self.init_lr = init_lr
        self.lr_step = init_lr / decay_end
        self.decay_start = decay_start
        

    def update(self, epoch: int):
        if epoch < self.decay_start:
            # do nothing
            return self.init_lr
        else:
            new_lr = self.init_lr - (self.lr_step * (epoch - self.decay_start))
            return new_lr


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
metric_keys = ['G_L1', 'G_GAN', 'G_total', 'D_real', 'D_fake', 'G_lr']
metrics_plt_pth = f'results/{dataset_name}/checkpoints/metrics.png'
n_samples = 400

lr = 0.0002
lr_beta1 = 0.5
lr_decay_start = 100
lr_decay_end = 100
lr_scheduler = LrScheduler(lr, lr_decay_start, lr_decay_end)
seed = np.random.randint(1, 10000)

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

check_generator = ImageDataGenerator(
    rescale=1./255,
    fill_mode='constant',
    data_format='channels_last',
    validation_split=0.0
)

check_loader = dataLoader(checkpoints_pth, check_generator, batch_sz=1, 
                          shuffle=False, img_sz=(256, 256), seed=seed)


# Optimizers
optimizer_d = Adam(lr=lr, beta_1=lr_beta1, beta_2=0.999, epsilon=1e-08, decay=0.0)
optimizer_g = Adam(lr=lr, beta_1=lr_beta1, beta_2=0.999, epsilon=1e-08, decay=0.0)

# Build and compile Discriminator
# ---------------------------------------------------------
inputs_dsc, outputs_dsc = build_discriminator(input_size=input_sz, 
                                              minibatch_std=minibatch_std)
discriminator = Model(inputs_dsc, outputs_dsc, name='discriminator')
discriminator.compile(loss=bce_loss, optimizer=optimizer_d, metrics=['accuracy'])
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

gan.compile(loss=[l1_loss, bce_loss], 
            loss_weights=[lambda_L1, 1.0], 
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
start_time = datetime.datetime.now()


# ganhack2: modified loss function/label flip real => 0
# label smoothing: real => 0.0 - 0.1
# real = np.random.random_sample((batch_size, ) + discriminator_output_sz) * 0.1 
# fake = np.ones((batch_size, ) + discriminator_output_sz)   # fake => 1

real = np.ones((batch_size, ) + discriminator_output_sz)   # real => 1
fake = np.zeros((batch_size, ) + discriminator_output_sz)  # fake => 0

for epoch in range(epochs):
    gen_checkpoint(gan, check_loader, epoch, checkpoints_pth)
    build_results_page(epoch)
    for batch in range(n_samples):                
        
        inputs, targets = next(train_loader)

        # Learning rate annealing
        # ----------------------------------------------
        new_lr = lr_scheduler.update(epoch)
        K.set_value(gan.optimizer.lr, new_lr)
        K.set_value(discriminator.optimizer.lr, new_lr)

        #  Train Generator
        # ----------------------------------------------
        # Train the generators in GAN setting
        g_loss = gan.train_on_batch([inputs], [targets, real])


        #  Train Discriminator
        # ----------------------------------------------
        outputs, _ = gan.predict(inputs)

        # Randomly switch the order to ensure discriminator isn't somehow 
        # memorising train order: Real, Fake, Real, Fake, Real, Fake
        if np.random.random() > 0.5:
            # Train the discriminators (original images = real / generated = Fake)
            d_loss_real, d_acc_real = discriminator.train_on_batch([targets, inputs], real)
            d_loss_fake, d_acc_fake = discriminator.train_on_batch([outputs, inputs], fake)
        else:
            d_loss_fake, d_acc_fake = discriminator.train_on_batch([outputs, inputs], fake)
            d_loss_real, d_acc_real = discriminator.train_on_batch([targets, inputs], real)

        # Plot the progress
        if (batch+1) % 100 == 0:
            elapsed_time = datetime.datetime.now() - start_time
            train_metrics.add({
                'epoch': epoch,
                'iters': batch+1,
                'G_lr': K.eval(gan.optimizer.lr),
                'D_lr': K.eval(discriminator.optimizer.lr),
                'G_L1': g_loss[1],
                'G_GAN': g_loss[2],
                'G_total': g_loss[0],
                'D_real': d_loss_real,
                'D_fake': d_loss_fake,
                'time': elapsed_time
        })

    train_metrics.to_csv()
    train_metrics.plot(metric_keys, metrics_plt_pth)
    # real_labels = np.zeros((1, ) + discriminator_output_sz) # no label smoothing at test time
    # evaluate_val(gan, discriminator, val_loader, real_labels, sample_dir, epoch, batch, experiment_title, val_metrics)

# Save models
gan.save(f'pretrained/{experiment_title}_gan.h5')
discriminator.save(f'pretrained/{experiment_title}_discriminator.h5')
