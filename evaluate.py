import numpy as np
import os
from skimage.transform import resize
from skimage.color import grey2rgb
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras.backend as K
from model.data_loader import normalize, denormalize
from model.discriminator_patchgan import bceWithLogitsLoss


def scale(x):
    return (x - x.min()) / x.max()

def sigmoid(x):
    return 1/(1+np.exp(-x))

def evaluate(gan, discriminator, data_loader, real, sample_dir, epoch, batch, experiment_title, metrics):
    """
    Sample 3 images from generator and save to sample_dir
    """
    # ensure output dir exists
    os.makedirs(sample_dir, exist_ok=True)

    inputs = []
    outputs_gen = []
    targets = []
    is_real = []
    patches = []
    for sample in range(3):
        # Run a forward pass
        # -----------------------
        input_, target = next(data_loader)
        
        output_gen, is_real_logit = gan.predict(input_)
        d_loss = discriminator.evaluate([output_gen, input_], real-1)
        g_loss = gan.evaluate(input_, [target, real])

        # Create heatmap from patchgan discriminator output
        # -----------------------
        # Apply sigmoid because model returns logits 
        d_activations = sigmoid(is_real_logit[0])    # TODO: consider best activation when other losses are used
        patch = resize(d_activations, (256, 256, 1), 
                       order=0, preserve_range=True, anti_aliasing=False) 
        patch = np.asarray(normalize(grey2rgb(patch[:, :, 0])))
        
        inputs += [input_]
        outputs_gen += [output_gen]
        targets += [target]
        is_real += [d_activations]
        patches += [patch]
 
        # Record metrics
        # -----------------------
        print('\nVALIDATION')
        metrics.add({
                'epoch': epoch,
                'batch': batch,
                'sample': sample,
                'D_loss': d_loss[0],
                'D_acc': d_loss[1],
                'G_total_loss': g_loss[0],
                'G_L1_loss': g_loss[1],
                'G_Disc_loss': g_loss[2]
        })
        print('\n')
        metrics.to_csv() 

    inputs = np.concatenate(inputs, axis=0)
    outputs_gen = np.concatenate(outputs_gen, axis=0)
    targets = np.concatenate(targets, axis=0)
    is_real = np.concatenate(is_real, axis=0)
    patches = np.stack(patches, axis=0)

    # Concat images
    imgs = np.concatenate([inputs, outputs_gen, patches, targets])
    imgs = denormalize(imgs)

    # Save sampled images to disk
    # -----------------------
    r, c = 4, 3
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
    fig.savefig(f'{sample_dir}/{str(epoch).zfill(4)}_{batch}_{experiment_title}.png')
    plt.close()
    return