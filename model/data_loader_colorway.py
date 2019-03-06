import numpy as np
import pandas as pd
from tensorflow import keras
import PIL

class DataGenerator(keras.utils.Sequence):
    """
    Generates data for Keras
    Adapted from: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    """
    def __init__(self, list_IDs, labels, colors_csv, batch_size=32, dim=(32,32), 
                 n_channels=1, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.colors_df = pd.read_csv(colors_csv, index_col='filename')
        self.on_epoch_end()
        self.n_batches = len(self.colors_df) // self.batch_size
        print(f'n_batches: {self.n_batches}')

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return np.array(X)/127.5 - 1, np.array(y)/127.5 - 1

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def parse_colors(self, ID):
        sample = self.colors_df.loc[ID].values
        colors = sample.reshape((-1, 3))
        return colors

    def gen_colorbar(self, colors, width, height):
        k = len(colors)
        sw = width // k
        colorbar = np.zeros((height, width, 3))
    
        # Populate colorbar with all but last color
        ones = np.ones((sw, height))
        for i in range(k-1):
            color = colors[i, :]
            rgb = np.asarray([ones * c for c in color]).T
            colorbar[:, i*sw:(i+1)*sw, :] = rgb
    
        # Populate colorbar with last color
        # This is a special case to ensure colorbar perfectly matches
        # image width, including any remainder pixels after dvision
        # of original image width by the number of classes k
        remainder = width - (sw*k)
        ones = np.ones((sw+remainder, height))
        i = k-1
        color = colors[i, :]
        rgb = np.asarray([ones * c for c in color]).T
        colorbar[:, i*sw:, :] = rgb
        
        return colorbar
    
    def add_colorbar(self, img_arr, colors, height):
        k = len(colors)
        width = img_arr.shape[1]
        
        # Get colorbar
        colorbar = self.gen_colorbar(colors, width, height)
        
        # Replace top edge of the target image with colorbar  
        img_arr[:colorbar.shape[0], :, :] = colorbar
        return img_arr 
    
    def greyscale(self, img_arr):
        bw = np.ones(img_arr.shape)
        for i in range(3):
            bw[:, :, i] = np.dot(img_arr, [0.2989, 0.587, 0.114]) 
        return bw
    
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.dim, self.n_channels))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store source image
            img = PIL.Image.open('data/processed/images/' + ID)
            img = img.resize(self.dim, resample=PIL.Image.BILINEAR).convert('RGB')
            img_arr = np.array(img)
            img_bw = self.greyscale(img_arr)
            colors = self.parse_colors(ID)
            colorbar_height = int(self.dim[0] // 20)
            X[i,] = self.add_colorbar(img_bw, colors, colorbar_height)

            # Store target image
            img_bw = self.add_colorbar(img_arr, colors, colorbar_height)
            y[i,] = img_bw

        return X, y