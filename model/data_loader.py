from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import matplotlib.pyplot as plt
import numpy as np

"""
Data Loader:
Loads the membrane cell segmentation dataset

Adapted and simplified from: 
https://github.com/zhixuhao/unet/blob/master/data.py
"""

def normalize(x):
    return (x - 0.5)*2

def denormalize(x):
    return (x/2)+0.5

def dataLoader(directory, data_generator, batch_sz=2, img_sz=(256, 256)):

    input_subdir = 'input'
    target_subdir = 'target'

    # Input generator
    x_gen = data_generator.flow_from_directory(
        directory,
        target_size=img_sz,
        color_mode="rgb",
        classes=[input_subdir],
        class_mode=None,
        batch_size=batch_sz,
        seed=1,
        interpolation='nearest'
    )
    
    # Target generator
    y_gen = data_generator.flow_from_directory(
        directory,
        target_size=img_sz,
        color_mode="rgb",
        classes=[target_subdir],
        class_mode=None,
        batch_size=batch_sz,
        seed=1,
        interpolation='nearest'
    )

    generator = zip(x_gen, y_gen)
    for (x, y) in generator:
        x, y = normalize(x), normalize(y)
        yield (x, y)


def show_augmentation(img_filepath, imageDataGenerator, n_rows=1):
    n_cols = 4
    img = load_img(img_filepath)
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape) 
    
    fig = plt.figure(figsize=(16, 8))
    i = 1
    for batch in imageDataGenerator.flow(x, batch_size=1, seed=1):
        ax = fig.add_subplot(n_rows, n_cols, i)
        ax.imshow(batch[0])
        ax.axis('off')
        i += 1
        if i > n_rows*n_cols: break
    plt.show();
    return


def show_sample(generator):
    batch = next(generator)
    x = denormalize(batch[0][0])
    y = denormalize(batch[1][0])
    
    size = (5, 5)
    plt.figure(figsize=size)
    plt.imshow(x)
    plt.show()
    plt.figure(figsize=size)
    plt.imshow(y)
    plt.show();
    return