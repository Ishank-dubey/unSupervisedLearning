import random
import numpy as np
from PIL import Image

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Normalize

from data_generation.image_classification import generate_dataset
from helpers import index_splitter, make_balanced_sampler
from beguinnersGuide2_1 import StepByStep


#'Convolution'
single = np.array(
    [[[[5, 0, 8, 7, 8, 1],
       [1, 9, 5, 0, 7, 7],
       [6, 0, 2, 4, 6, 6],
       [9, 7, 6, 6, 8, 4],
       [8, 3, 8, 5, 1, 3],
       [7, 2, 7, 0, 1, 0]]]]
)
print(single.shape)

identity = np.array(
    [[[[0, 0, 0],
       [0, 1, 0],
       [0, 0, 0]]]]
)
print(identity.shape)

region = single[:, :, 0:3, 0:3]
filtered_region = region * identity
total = filtered_region.sum()
print(total)

new_region = single[:, :, 0:3, (0+1):(3+1)]
print(new_region.sum())

# given an Image of hi x wi and filter hf x wf
# the convolution will result in the image iof size - (hi + 1 - hf, wi +1 - wf)
# this size is lesser than the image so
