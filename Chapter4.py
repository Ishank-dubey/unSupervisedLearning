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
#fig = plot_images(images, labels, n_plot=30)
#plt.show()

example = images[7]
#print(example)

example_hwc = np.transpose(example, (1, 2, 0))
#print(example_hwc)

tenzoriser = ToTensor()
example_from_hwc = tenzoriser(example_hwc)
example_from_chw = tenzoriser(example)
print(example_from_hwc)
print(example_from_chw)

transformed_image = ToPILImage()(example_from_hwc)
#plt.imshow(transformed_image, cmap='gray')

#Transformation like horizontal flip

flipper = RandomHorizontalFlip(p=1)
flipped_image = flipper(transformed_image)
#plt.imshow(flipped_image)
#plt.show()
print(flipped_image)


normalizer = Normalize(mean=(.5,), std=(.5,))
normalized_image = normalizer(tenzoriser(flipped_image))
print(normalized_image)#normalize accepts tensor image

#Compose
#print(labels)
composer = Compose([RandomHorizontalFlip(p=.5), Normalize(mean=(.5,), std=(.5,))])
composed_image = composer(tenzoriser(flipped_image))
print('ok')
print((composed_image == normalized_image).all())
x_tensor = torch.as_tensor(images/255).float()
y_tensor = torch.as_tensor(labels.reshape(-1,1)).float()

#Now lets create the data sets
class TransformTensorDataSet(Dataset):
    def __init__(self,x,y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform

    def __getitem__(self, index):
        x = self.x[index]
        if self.transform:
            x = self.transform(x)
        return x, self.y[index]
    def __len__(self):
        return len(self.x)


dataset = TransformTensorDataSet(x_tensor, y_tensor, composer)

def index_splitter(n, splits_weights, seed=13):
    idx = torch.arange(n)
    splits_tensor = torch.as_tensor(splits_weights)
    multiplier = n / splits_tensor.sum()
    splits_tensor = (multiplier * splits_tensor).long()
    difference = n - splits_tensor.sum()
    splits_tensor[0] += difference
    torch.manual_seed(seed)
    return random_split(idx, splits_tensor)

index_splitter(11, [80, 20])

train_idx, val_idx = index_splitter(len(x_tensor),[80, 20])
train_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(val_idx)

train_loader = DataLoader(dataset=dataset, batch_size=16, sampler=train_sampler)
val_loader = DataLoader(dataset=dataset, batch_size=16, sampler=val_sampler)
print(len(iter(train_loader)))
print(len(iter(val_loader)))

#now when we want to perform the augumentation, then we will need apply the composer1 on the train dataset and
#composer 2 to the validaiton data set
#here the sampler wont be used as we have to have two data sets anyways
# in the DataLoader we can give shuffel as true and seperate datasets

x_train_tensor = x_tensor[train_idx]
y_train_tensor = y_tensor[train_idx]

x_val_tensor = x_tensor[val_idx]
y_val_tensor = y_tensor[val_idx]

train_composer = Compose([RandomHorizontalFlip(p=.5), Normalize(mean=(.5, ), std=(.5,))])
val_composer = Compose([Normalize(mean=(.5, 0), std=(.5,))])

#create the seperate datasets
train_dataset = TransformTensorDataSet(x_train_tensor, y_train_tensor, train_composer)
val_dataset = TransformTensorDataSet(x_val_tensor, y_val_tensor, val_composer)

train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=16)
val_dataset = DataLoader(dataset=val_dataset, batch_size=16)

#we also need to see if the training set is imbalanced
#i.e. how many are positive and how many are negative in the training set
classes, count = y_train_tensor.unique(return_counts=True)
#print(classes, count)
weights = 1 / count
print(weights)

sample_weights = weights[y_train_tensor.squeeze().to(torch.int64)]
generator = torch.Generator()
sampler = WeightedRandomSampler(
    num_samples=len(sample_weights),
    weights = sample_weights,
    replacement=True,
    generator = generator
)


train_loader = DataLoader(
    sampler=sampler,
    batch_size=16,
    dataset=train_dataset
)
train_loader.sampler.generator.manual_seed(42)
random.seed(42)
sumSampler = 0
for t in train_loader:
    sumSampler += t[1].sum()

print(sumSampler)


def make_sampler(y):
    classes, count = y.unique(return_count=True)
    weights = 1 / count
    sample_weights = weights[y.squeeze().to(torch.int64)]
    generator = torch.Generator()
    sampler = WeightedRandomSampler(
        weights = sample_weights,
        num_samples=len(sample_weights),
        generator=generator,
        replacement=True
    )
    return sampler


#Now we have the data prepared - idx, train_idx, val_idx, Transformed Data, x_train_tensor, y_train_tensor
#As of now we going to flaten the 2D images into flat arrays using the nn.Flaten

sample_test = next(iter(train_loader))
print(sample_test[0])
print(sample_test[0].shape)


#Now its a linear modal
lr = .1
torch.manual_seed(17)
modal = nn.Sequential()
modal.add_module('flatten', nn.Flatten())
modal.add_module('hidden0', nn.Linear(25, 5, bias=False))
modal.add_module('hidden1', nn.Linear(5, 3, bias=False))
modal.add_module('hidden2', nn.Linear(3, 1, bias=False))
modal.add_module('sigmoid', nn.Sigmoid())
optimizer = optim.SGD(modal.parameters(), lr=lr)
loss_fn = nn.BCELoss()

sbs_nn = StepByStep(modal, loss_fn, optimizer )
sbs_nn.set_loaders(train_loader, val_loader)
sbs_nn.train(100)
fig = sbs_nn.plot_losses()
#fig.plot()