import os
import numpy as np
import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import numpy as np

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
        if os.path.exists(logs_pth):
            os.remove(logs_pth)
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


    def plot(self, metrics: list, out_pth: str):
        """ Plot training metrics in grid """
        rows, cols = 2, 3
        df = pd.read_csv(self.logs_pth)

        fig, axs = plt.subplots(rows, cols, figsize=(15, 8))
        for i in range(len(metrics)):
            ax = axs[i//cols, i%cols]
            sns.lineplot(x=list(df.index), y=metrics[i], data=df, ax=ax)
            sns.regplot(x=list(df.index), y=metrics[i], data=df, scatter=False, 
                        lowess=True, ax=ax)
        fig.savefig(out_pth)
        # close all plots to avoid memory warning
        plt.close('all')


def print_setup(tf_version, tf_eager, args, d_count, g_count, gan_count):
    """
    Parse and print setup details
    """
    print('\n-------------- Tensorflow --------------- \n')

    print(f'{"TensorFlow version": >20}: {tf_version: <}')
    print(f'{"Eager execution": >20}: {tf_eager: <}')

    print('\n---------------- Options ---------------- \n')
    for arg in vars(args):
        print(f'{arg: >20}: {getattr(args, arg): <}')

    print('\n---------- Trainable Parameters ---------- \n')
    print(f'{"discriminator": >20}: {d_count: <}')
    print(f'{"generator": >20}: {g_count: <}')
    print(f'{"gan": >20}: {gan_count: <}')
    print('\n------------------------------------------ \n')