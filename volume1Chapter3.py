import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
from StepByStep import StepByStep
from util3 import *
X, y = make_moons(n_samples=100, noise=.3, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.2, random_state=13)
sc = StandardScaler()
sc.fit(X_train)
sc.transform(X_train)
sc.transform(X_val)

figure1(X_train, y_train, X_val, y_val)

x_train_tensor = torch.as_tensor(X_train).float()
y_train_tensor = torch.as_tensor(y_train.reshape(-1, 1)).float()

x_val_tensor = torch.as_tensor(X_val).float()
y_val_tensor = torch.as_tensor(y_val.reshape(-1, 1)).float()

train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
val_dataset = TensorDataset(x_val_tensor, y_val_tensor)

train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=16)
val_loader = DataLoader(dataset=val_dataset, batch_size=16)

#figure2(.25)
#figure3(.25)

"""
p = 1/(1+e^-z) = logistic regression
"""
def odds_ratio(prob1):
    return prob1/(1 - prob1)
def log_odds_ratio(prob):
    return np.log(odds_ratio(prob))

def sigmoid(z):
    return 1/(np.exp(-z) + 1)

print(torch.sigmoid(torch.tensor(1.0986)))
#print(torch.nn.Sigmoid(input=torch.tensor(1.0986)))
print(sigmoid(1.0986))

torch.manual_seed(42)
model1 = nn.Sequential()
model1.add_module('linear', nn.Linear(2, 1))
model1.add_module('sigmoid', nn.Sigmoid())
print(model1.state_dict()) #Sigmoid is a model but there is no learnable parameters inside it.

#Model Confuguration
lr = .1
model = nn.Sequential()
linear = nn.Linear(2, 1)
model.add_module('linear', linear)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=lr)
n_epochs = 100

sbs = StepByStep(model, loss_fn, optimizer)
sbs.set_loaders(train_loader, val_loader)
sbs.train(n_epochs)
fig = sbs.plot_losses()
print(model.state_dict())

predictions = sbs.predict(x_train_tensor[:4])
print(predictions)

probabilities = sigmoid(predictions)
print(probabilities)

"""
sigmoid(z) >=.5 when the z >= 0m when the z < 0 then the sigmoid(z) will be less than .5
So we can use the prediction i.e. logit directly
"""
classes = (predictions >= 0).astype(int)
print(classes)
