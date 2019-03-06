import numpy as np
import os
from skimage.transform import resize
from skimage.color import grey2rgb
import matplotlib.pyplot as plt

import tensorflow as tf

from model.data_loader import normalize, denormalize


def sample_images(gan, data_loader, sample_dir, 
                  epoch, batch, experiment_title):
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
    
    outputs, is_real = gan.predict(inputs)

    # Create heatmap from patchgan discriminator output
    # -----------------------
    print(f'Patch Size: {is_real[0].shape}')
    patches = [resize(x, (256, 256, 1), order=0, preserve_range=True, anti_aliasing=False) 
               for x in is_real]
    patches = [1-normalize(grey2rgb(p[:, :, 0])) for p in patches]
    patches = np.asarray(patches)

    # Stack images
    imgs = np.concatenate([inputs, outputs, patches, targets])
    imgs = denormalize(imgs)

    # Save images to disk
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