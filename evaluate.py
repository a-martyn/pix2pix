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
    Numpy array to png file on disk
    """
    arr = np.asarray(denormalize(arr)*255, dtype='uint8')
    img = Image.fromarray(arr)
    img.save(filepath)
    return


def gen_checkpoint(gan, check_loader, epoch, output_pth):
    """
    Generate prediction from training set for comparison with
    authors checkpoints from pytorch implementation
    """
    inputs, targets = next(check_loader)
    outputs, d_activations = gan.predict(inputs)
    patch = patchgan_heatmap(d_activations[0])
    
    arr2png(outputs[0], f'{output_pth}/gen_tf/{str(epoch).zfill(4)}')
    arr2png(patch, f'{output_pth}/patch_tf/{str(epoch).zfill(4)}')
    return

# def plot_results(imgs, is_real, sample_dir, experiment_title):
#     """
#     Save sampled images to disk
#     """
#     r, c = 4, 3
#     titles = ['Condition', 'Generated', 'Patches', 'Original']
#     fig, axs = plt.subplots(r, c, figsize=(10, 10))
#     cnt = 0
#     for i in range(r):
#         for j in range(c):
#             axs[i,j].imshow(imgs[cnt])
#             axs[i,j].axis('off')
#             if i == 2:
#                 axs[i,j].set_title(f'{titles[i]}: {np.mean(is_real[j]): .2f}')
#             else:
#                 axs[i, j].set_title(titles[i])
#             cnt += 1
#     save_freq = 10
#     fig.savefig(f'{sample_dir}/{str((epoch//save_freq)*save_freq).zfill(4)}_{experiment_title}.png')
#     fig.savefig(f'{sample_dir}/_latest_{experiment_title}.png')
#     plt.close()


# def evaluate(gan, discriminator, data_loader, real, sample_dir, epoch, batch, experiment_title, metrics):
#     """
#     Sample 3 images from generator and save to sample_dir
#     """
#     # ensure output dir exists
#     os.makedirs(sample_dir, exist_ok=True)

#     inputs = []
#     outputs_gen = []
#     targets = []
#     is_real = []
#     patches = []
    
#     print('\nVALIDATION')
#     for sample in range(3):
#         # Run a forward pass
#         # -----------------------
#         input_, target = next(data_loader)
        
#         output_gen, d_activations = gan.predict(input_)
#         d_loss = discriminator.evaluate([output_gen, input_], real+1, verbose=0)
#         g_loss = gan.evaluate(input_, [target, real], verbose=0)

#         # Create heatmap from patchgan discriminator output
#         # -----------------------
#         patch = patchgan_heatmap(d_activations)
        
#         inputs += [input_]
#         outputs_gen += [output_gen]
#         targets += [target]
#         is_real += [d_activations]
#         patches += [patch]
 
#         # Record metrics
#         # -----------------------
#         metrics.add({
#                 'epoch': epoch,
#                 'batch': batch,
#                 'sample': sample,
#                 'D_loss': d_loss[0],
#                 'D_acc': d_loss[1],
#                 'G_total_loss': g_loss[0],
#                 'G_L1_loss': g_loss[1],
#                 'G_Disc_loss': g_loss[2]
#         })
#         metrics.to_csv() 
#     print('\n')

#     inputs = np.concatenate(inputs, axis=0)
#     outputs_gen = np.concatenate(outputs_gen, axis=0)
#     targets = np.concatenate(targets, axis=0)
#     is_real = np.concatenate(is_real, axis=0)
#     patches = normalize(np.stack(patches, axis=0))

#     # Concat images
#     imgs = np.concatenate([inputs, outputs_gen, patches, targets])
#     imgs = denormalize(imgs)

#     plot_results(imgs, is_real, sample_dir, experiment_title)
#     return
