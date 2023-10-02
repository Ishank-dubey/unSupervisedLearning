import os
import sys
import errno
import requests
import subprocess
import shutil
from IPython.display import HTML, display
from tensorboard import manager


def download_to_colab(branch='master'):
    base_url = 'https://raw.githubusercontent.com/Ishank-dubey/unSupervisedLearning/master/'

    folders = ['data_generation', 'images', 'plots', 'runs']
    for folder in folders:
        print('IN')
        if len(folder):
            try:
                os.mkdir(folder)
            except OSError as e:
                print('RAISe')
                if e.errno != errno.EEXIST:
                    raise
    path = os.path.join('data_generation', 'image_classification.py')
    url = '{}{}'.format(base_url, path)
    r = requests.get(url, allow_redirects=True)
    open(path, 'wb').write(r.content)

    path = os.path.join('StepByStep.py')
    url = '{}{}'.format(base_url, path)
    r = requests.get(url, allow_redirects=True)
    open(path, 'wb').write(r.content)

    path = os.path.join('helpers.py')
    url = '{}{}'.format(base_url, path)
    r = requests.get(url, allow_redirects=True)
    open(path, 'wb').write(r.content)

    path = os.path.join('plots', 'chapter5.py')
    url = '{}{}'.format(base_url, path)
    r = requests.get(url, allow_redirects=True)
    open(path, 'wb').write(r.content)

##download_to_colab()




# This is needed to render the plots in this chapter
from plots.chapter5 import *