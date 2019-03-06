from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import cv2
from sklearn.cluster import KMeans
from fastprogress import master_bar, progress_bar


class DominantColors:
    """
    Get dominant colors of a batch of images.
    Record to csv.
    Parallel processing
    """
    
    def __init__(self, k=5):
        self.k = k
        
        # Create an empty dataframe
        cols = ['filename']
        cols = []
        for i in range(5):
            rgb = [f'r{i}', f'g{i}', f'b{i}']
            cols += rgb
        self.colors_df = pd.DataFrame(columns=cols) 
    
    
    def record_colors(self, filename, colors):
        """
        Add record to dataframe
        note: dataframe is held out of scope for parallelisation
        """
        try:
            fn = {'filename': filename}
            rs = {f'r{i}': v[0] for i, v in enumerate(colors)}
            gs = {f'g{i}': v[1] for i, v in enumerate(colors)}
            bs = {f'b{i}': v[2] for i, v in enumerate(colors)}
            sample = {**fn, **rs, **gs, **bs}
            self.colors_df = self.colors_df.append(sample, ignore_index=True)
        except:
            print(f'Error recording: {filename}')
        return
    
    def to_csv(self, filepath):
        self.colors_df.to_csv(filepath, index=False)
        return
    
    def analyse(self, filepath):
        """
        Analyse and record dominant colors of image at given filepath
        Unused argument is for compatibility eith fastai's parallel()
        function.
        """
        _, filename = os.path.split(filepath)
        
        try:
            #read image
            img = cv2.imread(filepath)
            
            #convert to rgb from bgr
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
            #reshaping to a list of pixels
            img = img.reshape((img.shape[0] * img.shape[1], 3))
            
            #using k-means to cluster pixels
            kmeans = KMeans(n_clusters=self.k, precompute_distances=True)
            kmeans.fit(img)
            
            #the cluster centers are our dominant colors.
            colors = kmeans.cluster_centers_
        
            # Sort color clusters by decreasing population
            # show that most dominant color is returned first
            labels = kmeans.labels_
            unique, counts = np.unique(labels, return_counts=True)
            colors_sorted = sorted(zip(counts, colors), reverse=True)
            colors_sorted = [c[1] for c in colors_sorted]
            colors_sorted = np.asarray(colors_sorted)
            
            return (filename, colors_sorted)
        
        except Exception as e:
            print(f'Error processing: {filename}')
            print(e)
            return (filename, [np.nan for _ in range(self.k)])

    def analyse_batch(self, filepaths, csv_path, chunksize=1):
        """
        Analyse images in parallel using Pythons pool exectution.
        Apparently 'large' chunks sizes are faster when len filepaths is large
        """
        count = 0
        for fp in progress_bar(filepaths):
            filename, colors = self.analyse(fp)
            self.record_colors(filename, colors)
            if count % 500 == 0:
                self.to_csv(csv_path)




def parallel(func, arr, max_workers:int=8):
    "Call `func` on every element of `arr` in parallel using `max_workers`."
    if max_workers<2: _ = [func(o,i) for i,o in enumerate(arr)]
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(func,o,i) for i,o in enumerate(arr)]
            for f in progress_bar(as_completed(futures), total=len(arr)): pass


class Metrics():
    """
    Display and record metrics in ML training loop. Log 
    metrics to whilst minimising memory footprint and minimizing
    disk write time.

    Example Usage:
    1. `metrics = Metrics('path/to/logs.csv')`
    2. `metrics.add({'epoch': 1, 'batch': 24, 'D_loss': 0.253678903})`
    3. repeat step 2 in training loop then call: `metrics.to_csv()` to
       append cached metrics to csv file
    """

    def __init__(self, logs_pth):
        self.logs_pth = logs_pth   # output csv filepath
        self.cache = None

    def to_cache(self, metrics: dict):
        """
        Cache metrics until `.to_csv` is called
        """
        if not self.cache:
            self.cache  = {k: [v] for k, v in metrics.items()}
        else:
            for k, v in metrics.items():
                self.cache[k].append(v)
        return

    def print(self, metrics: dict):
        """
        Print metrics to command line with formatting
        """
        def format(value):
            if isinstance(value, (float, np.float32, np.float64)):
                return f'{value: .5f}'
            else:
                return str(value)
        stringified = [f'{k}: {format(v)}' for k, v in metrics.items()] 
        print(' | '.join(stringified))
        return
   
    def to_csv(self):
        """
        Append cached metrics to csv file at `self.logs_pth`
        - Create file unless exists, otherwise append
        - Add header if file is being created, otherwise skip it
        """
        df = pd.DataFrame(self.cache)
        with open(self.logs_pth, 'a') as f:
            df.to_csv(f, header=f.tell()==0)
        self.cache = None
        return

    def add(self, metrics: dict):
        """
        Add metrics to cache and print to command line
        """
        self.to_cache(metrics)
        self.print(metrics)
        return



