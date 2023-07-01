import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn as nn
import torch.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

def figure1(x, y):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    ax.scatter(x, y)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_ylim([0, 3.1])
    ax.set_title('Generated Data - Full Dataset')
    fig.tight_layout()
    plt.show()
def prepare_data(x, y):
    torch.manual_seed(13)

    # Builds tensors from numpy arrays BEFORE split
    x_tensor = torch.as_tensor(x).float()
    y_tensor = torch.as_tensor(y).float()

    # Builds dataset containing ALL data points
    dataset = TensorDataset(x_tensor, y_tensor)

    # Performs the split
    ratio = .8
    n_total = len(dataset)
    n_train = int(n_total * ratio)
    n_val = n_total - n_train

    train_data, val_data = random_split(dataset, [n_train, n_val])

    # Builds a loader of each set
    train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=16)
    return train_loader, val_loader
def modal_configuration():
    lr = .1
    torch.manual_seed(42)
    modal = torch.nn.Sequential(nn.Linear(1, 1))
    optimizer = torch.optim.SGD(modal.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss(reduction='mean')
    return modal, optimizer, loss_fn
