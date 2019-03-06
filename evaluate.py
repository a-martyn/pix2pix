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

def evaluate(gan, discriminator, data_loader, y_true_fake, sample_dir, epoch, batch, experiment_title, metrics):
    """
    Sample 3 images from generator and save to sample_dir
    """
    # ensure output dir exists
    os.makedirs(sample_dir, exist_ok=True)

    # Run a forward pass
    # -----------------------
    inputs, targets = next(data_loader)
    # because last batch in epoch can be smaller, skip it
    if inputs.shape[0] != 3:
        inputs, targets = next(data_loader)
    
    outputs_gen, is_real_logit = gan.predict(inputs)
    d_loss = discriminator.evaluate([outputs_gen, inputs], y_true_fake)
    g_loss = gan.evaluate(inputs, [targets, y_true_fake])

    # Record metrics
    # -----------------------
    print('\nVALIDATION')
    metrics.add({
            'epoch': epoch,
            'batch': batch,
            'D_loss': d_loss[0],
            'D_acc': d_loss[1],
            'G_total_loss': g_loss[0],
            'G_L1_loss': g_loss[1],
            'G_Disc_loss': g_loss[2]
    })
    print('\n')
    metrics.to_csv()    
    
    # Create heatmap from patchgan discriminator output
    # -----------------------
    
    #print(f'Patch Size: {is_real_logit[0].shape}')
    
    # Apply sigmoid because model returns logits 
    # TODO: consider bes activation when other losses are used
    is_real = [sigmoid(x) for x in is_real_logit]
    patches = [resize(x, (256, 256, 1), order=0, preserve_range=True, anti_aliasing=False) 
               for x in is_real]
    patches = [normalize(grey2rgb(p[:, :, 0])) for p in patches]
    
    patches = np.asarray(patches)

    # Stack images
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