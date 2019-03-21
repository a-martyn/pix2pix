import dominate
from dominate.tags import meta, h3, table, tr, td, p, a, img, br
import os

"""
Generates html for training visualisation web page.

Adapted from:
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/util/html.py
"""


class HTML:
    """This HTML class allows us to save images and write texts into a single HTML file.
     It consists of functions such as <add_header> (add a text header to the HTML file),
     <add_images> (add a row of images to the HTML file), and <save> (save the HTML to the disk).
     It is based on Python library 'dominate', a Python library for creating and manipulating HTML documents using a DOM API.
    """

    def __init__(self, img_dir, title, refresh=0):
        """Initialize the HTML classes
        Parameters:
            web_dir (str) -- a directory that stores the webpage. HTML file will be created at <web_dir>/index.html; images will be saved at <web_dir/images/
            title (str)   -- the webpage name
            refresh (int) -- how often the website refresh itself; if 0; no refreshing
        """
        self.title = title
        self.img_dir = img_dir
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)

        self.doc = dominate.document(title=title)
        if refresh > 0:
            with self.doc.head:
                meta(http_equiv="refresh", content=str(refresh))

    def get_image_dir(self):
        """Return the directory that stores images"""
        return self.img_dir

    def add_header(self, text, im):
        """Insert a header to the HTML file
        Parameters:
            text (str) -- the header text
        """
        with self.doc:
            h3(text)
            with a(href=im):
                img(style="width:1024px", src=im)

    def add_images(self, ims, txts, links, epoch, width=256):
        """add images to the HTML file
        Parameters:
            ims (str list)   -- a list of image paths
            txts (str list)  -- a list of image names shown on the website
            links (str list) --  a list of hyperref links; when you click an image, it will redirect you to a new page
        """
        self.t = table(border=1, style="table-layout: fixed;")  # Insert a table
        self.doc.add(self.t)
        with self.t:
            with tr():
                for im, txt, link in zip(ims, txts, links):
                    with td(style="word-wrap: break-word;", halign="center", valign="top"):
                        with p():
                            with a(href=link):
                                img(src=im)
                            br()
                            p(txt)
            p(epoch)

    def save(self, filepath: str):
        """save the current content to the HMTL file"""
        f = open(filepath, 'wt')
        f.write(self.doc.render())
        f.close()


def build_results_page(epochs:int, checkpoints_pth:str,
                      checkpoint_dir_labels:dict, metrics_plt_filepath:str, 
                      html_filepath:str):
    """
    Generate html for training visualisation webpage. E.g. call on each epoch 
    to show progress.

    - epochs: current epoch, determines how many epochs to show results for
    - checkpoints_pth: path to directory where checkpoint images are stored
    - metrics_plt_filename: filename for metrics plot
    - checkpoint_dir_labels: directory names an labels for checkpoint images
                             expected at {checkpoints_pth}/images
    """
    html = HTML(checkpoints_pth, 'test_html')
    html.add_header('Training results', f'../{metrics_plt_filepath}')

    dirs   = list(checkpoint_dir_labels.keys())
    labels = list(checkpoint_dir_labels.values())

    for n in range(epochs, -1, -1):
        fn = f'{str(n).zfill(4)}.png'
        ims, txts, links, epoch = [], [] , [], []
        for d in zip(dirs, labels):
            ims.append(f'../{checkpoints_pth}/{d[0]}/{fn}')
            txts.append(f'{d[1]}')
            links.append(f'../{checkpoints_pth}/{d[0]}/{fn}')
        epoch.append(f'epoch: {n}')
        html.add_images(ims, txts, links, epoch)
    html.save(html_filepath)


if __name__ == '__main__':
    results_pth = 'results'
    checkpoints_pth = 'data/facades_processed/checkpoints'
    metrics_plt_filepath = f'{results_pth}/metrics.png'
    html_filepath = f'{results_pth}/index.html'
    checkpoint_dir_labels = {
        'input': 'input', 
        'gen_pytorch': "authors' pytorch", 
        'gen_tf': 'this implementation', 
        'target': 'target', 
        'patch_tf': 'patchgan'
    }
    build_results_page(199, checkpoints_pth, checkpoint_dir_labels, 
                       metrics_plt_filepath, html_filepath)
