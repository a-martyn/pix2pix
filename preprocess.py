import os
import shutil
import numpy as np
from PIL import Image

"""
Preprocessing script for facades dataset.

Separates combined 512x256 input target images into
two 256x256 images in distinct directories
"""


src_pth = 'data/facades'
out_pth = 'data/facades_processed'
dirs = ['train', 'val', 'test']

for d in dirs:
    # Make target directory structure
    os.mkdir(f'{out_pth}/{d}')
    os.mkdir(f'{out_pth}/{d}/input')
    os.mkdir(f'{out_pth}/{d}/target')
    filenames = os.listdir(f'{src_pth}/{d}')
    for fn in filenames:
        # Split combined images into input and targets
        # in separate dirs
        print(f'PROCESSING: {fn}')
        img = Image.open(f'{src_pth}/{d}/{fn}')
        arr = np.asarray(img)
        w = arr.shape[1]
        input_ = Image.fromarray(arr[:, w//2:, :])
        target = Image.fromarray(arr[:, :w//2, :])
        input_.save(f'{out_pth}/{d}/input/{fn}')
        target.save(f'{out_pth}/{d}/target/{fn}')

print('DONE')