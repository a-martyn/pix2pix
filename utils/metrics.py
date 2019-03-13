import numpy as np
import pandas as pd
import os


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


    def plot(self, metrics: list, out_pth: str):
        """ Plot training metrics in grid """
        rows, cols = 2, 3
        df = pd.read_csv(self.logs_pth)
        # filter to last iter in epoch only
        df_filt = df[df['iters'] == 300][5:]

        fig, axs = plt.subplots(rows, cols, figsize=(15, 8))
        for i in range(len(metrics)):
            ax = axs[i//cols, i%cols]
            sns.lineplot(x='epoch', y=metrics[i], data=df_filt, ax=ax)
            sns.regplot(x='epoch', y=metrics[i], data=df_filt, scatter=False, lowess=True, ax=ax)
        
        fig.savefig(out_pth)
