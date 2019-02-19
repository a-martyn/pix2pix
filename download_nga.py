#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataSciece.changeDirOnImportExport setting
import os
try:
    os.chdir(os.path.join(os.getcwd(), 'colorway'))
    print(os.getcwd())
except:
    pass

#%%
import requests
from bs4 import BeautifulSoup
from pprint import pprint
import shutil
import json
import numpy as np
import time
from tqdm import tqdm


#%%
# Declare functions

def get_html(urls:list):    
    """ Get raw html for each url listed """
    htmls = []
    for url in urls:
        r = requests.get(url)
        html = r.text
        htmls.append(html)
    return htmls


def parse(htmls:list):
    """ Parse asset ids from html """
    asset_ids = []
    for html in htmls:
        soup = BeautifulSoup(html, 'html.parser')
        imgs = soup.find_all("a", class_="imageLink")
        asset_ids += [i['assetid'] for i in imgs]
    return asset_ids


def download(asset_ids:list, output_path:str):
    size = 2  # this returns a 1200 px image, but larger should be available somehow
    api = 'https://images.nga.gov/'
    
    for asset_id in asset_ids:
        time.sleep(np.random.random()*1)
        url = f'{api}?service=asset&action=download_comp_image&asset={asset_id}&size={size}'
        path = f'{output_path}{asset_id}.jpg'
        print(f'downloading: {asset_id}')
        try:
            r = requests.get(url, stream=True)
            if r.status_code == 200:
                with open(path, 'wb') as f:
                    for chunk in r:
                        f.write(chunk)
            else:
                print(f'error: {r.status_code}')
        except:
            print('error: download')


#%%
# Set filters

# Painting (2301 images)
pages = 31*3
painting_urls = [f"https://images.nga.gov/en/search/show_advanced_search_page.html?service=search&action=do_advanced_search&language=en&form_name=default&all_words=&exact_phrase=&exclude_words=&artist_last_name=&keywords_in_title=&accession_number=&school=&Classification=&medium=&year=&year2=&qw=%22Open%20Access%20Available%22&page={page+1}&qw=%22Open+Access+Available%22%20and%20%22Painting%22"
                 for page in range(pages)]

# search Textile (102 images)
pages = 5
textile_urls = [f'https://images.nga.gov/?service=search&action=do_quick_search&language=en&mode=&q=textile&qw=&mime_type=&page={page+1}&grid_layout=1&grid_thumb=7'
                for page in range(pages)]

categories = {
    'painting': painting_urls,
    'textile': textile_urls
}
for k in categories:
    output_path = f'./data/nga_{k}/'
    print('Getting html')
    htmls = get_html(categories[k])
    print('Parsing asset ids')
    asset_ids = parse(htmls)
    print('Downloading')
    download(asset_ids, output_path)
