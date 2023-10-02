import os
import sys
import errno
import requests
import subprocess
import shutil
from IPython.display import HTML, display
from tensorboard import manager

def tensorboard_cleanup():
    info_dir = manager._get_info_dir()
    shutil.rmtree(info_dir)
FOLDERS = {
    0: ['plots'],
    1: ['plots'],
    2: ['plots', 'data_generation', 'data_preparation', 'model_configuration', 'model_training'],
    21: ['plots', 'data_generation', 'data_preparation', 'model_configuration', 'stepbystep'],
    3: ['plots', 'stepbystep'],
    4: ['plots', 'stepbystep', 'data_generation'],
    5: ['plots', 'stepbystep', 'data_generation', ''],
    6: ['plots', 'stepbystep', 'stepbystep', 'data_generation', 'data_generation', 'data_preparation'],
    7: ['plots', 'stepbystep', 'data_generation'],
    71: ['plots', 'stepbystep', 'data_generation'],
    8: ['plots', 'plots', 'stepbystep', 'data_generation'],
    9: ['plots', 'plots', 'plots', 'stepbystep', 'data_generation'],
    10: ['plots', 'plots', 'plots', 'plots', 'stepbystep', 'data_generation', 'data_generation', '', ''],
    11: ['plots', 'stepbystep', 'data_generation', ''],
}