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

def get_html(url:str):    
    """ Get raw html for every page at given url """
    page = 1
    htmls = []
    status = 200
    while status != 500:
        try:
            r = requests.get(url.format(page))
            response = r.json()
            html = response['html']
            htmls.append(html)
            print('html downloaded')
        except:
            print('error: html')
        status = r.status_code
        page += 1
    return htmls


def parse(htmls:list):
    """ Parse asset ids from html """
    urls = []
    for html in htmls:
        soup = BeautifulSoup(html, 'html.parser')
        imgs = soup.find_all("img")
        urls += [i['data-iiifid'] for i in imgs]
    return urls


def set_size(urls:list, width:int):
    new = []
    for url in urls:
        stub = f'/full/{width},/0/default.jpg'
        asset_id = url[29:] + '_' + str(width)
        result = (asset_id, url+stub)
        new.append(result)
    return new

        

def download(assets:list, output_path:str):

    for asset in assets:
        asset_id, url = asset
        time.sleep(np.random.random()*1)
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
# categories = {
#     'textile':   'https://www.artic.edu/collection/more?is_public_domain=1&classification_ids=textile&page={}',
#     'painting':  'https://www.artic.edu/collection/more?is_public_domain=1&classification_ids=painting&page={}',
#     'hiroshige': 'https://www.artic.edu/collection/more?is_public_domain=1&artist_ids=Utagawa+Hiroshige&page={}'
# }
categories = {
    'paper_fiber':   'https://www.artic.edu/collection/more?is_public_domain=1&material_ids=paper+(fiber+product)&page={}'
}

for k in categories:
    output_path = f'./data/artic_{k}/'
    print('Getting html')
    htmls = get_html(categories[k])
    print('Parsing asset ids')
    urls = parse(htmls)
    sized_urls = set_size(urls, 1024)
    print('Downloading')
    download(sized_urls, output_path)


