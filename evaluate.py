import numpy as np
import os
from skimage.transform import resize
from skimage.color import grey2rgb
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
import tensorflow.keras.backend as K
from model.data_loader import normalize, denormalize


def patchgan_heatmap(d_activations):
    """
    Create heatmap from patchgan discriminator output
    """
    patch = resize(d_activations, (256, 256, 1), 
                   order=0, preserve_range=True, anti_aliasing=False, mode='constant') 
    patch = np.asarray(grey2rgb(patch[:, :, 0]))
    return patch


def arr2png(arr, filepath: str):
    """
    Numpy array to png image on disk
    """
    arr = np.asarray(denormalize(arr)*255, dtype='uint8')
    img = Image.fromarray(arr)
    img.save(filepath)
    return


def gen_checkpoint(gan, check_loader, epoch, output_pth):
    """
    Generate prediction from training set for comparison with
    authors' own checkpoints from their pytorch implementation
    """
    inputs, targets = next(check_loader)
    outputs, d_activations = gan.predict(inputs)
    patch = patchgan_heatmap(d_activations[0])
    
    arr2png(outputs[0], f'{output_pth}/gen_tf/{str(epoch).zfill(4)}.png')
    arr2png(patch, f'{output_pth}/patch_tf/{str(epoch).zfill(4)}.png')
    return

