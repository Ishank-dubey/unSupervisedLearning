import random
import numpy as np
from PIL import Image
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split, WeightedRandomSampler, SubsetRandomSampler
from torchvision.transforms import Compose, ToTensor, Normalize, ToPILImage, RandomHorizontalFlip, Resize

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
#%matplotlib inline

from data_generation.image_classification import generate_dataset
from beguinnersGuide2_1 import StepByStep
from plots.chapter4 import *
images, labels = generate_dataset(img_size=5, n_images=300, binary=True, seed=13)
fig = plot_images(images, labels, n_plot=30)
plt.show()
