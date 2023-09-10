##Going classy##

import numpy as np
import datetime
import torch
import torch.optim as optim
import torch.nn as nn
import torch.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from util2 import *
print(DataLoader)
print(TensorDataset)

class StepByStep(object):
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.train_loader = None
        self.val_loader = None
        self.writer = None
        self.val_losses = []
        self.total_epochs = 0
        self.losses = []
        self.train_step_fn = self._make_train_step_fn()
        self.val_step_fn = self._make_val_step_fn()
    def to(self, device):
        try:
            self.device = device
            self.model.to(device)
        except RuntimeError:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"Couldn't send model to {device}, sending to {self.device} instead")
            self.model.to(self.device)
    def set_loaders(self, train_loader, val_loader=None):
        #Let the train and validation loaders be loaded later
        self.train_loader = train_loader
        self.val_loader = val_loader
    def set_tensorboard(self, name, folder='runs'):
        suffix = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        self.writer = SummaryWriter(f'{folder}/{name}_{suffix}')

    def _make_train_step_fn(self):
        def perform_train_step_fn(x, y):
            self.model.train()
            yhat = self.model(x)
            loss = self.loss_fn(yhat, y)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            return loss.item()
        return perform_train_step_fn

    def _make_val_step_fn(self):
        def perform_val_step_fn(x, y):
            self.model.eval()
            yhat = self.model(x)
            loss = self.loss_fn(yhat, y)
            return loss.item()
        return perform_val_step_fn
    def _mini_batch(self, validation=False):
        if validation:
            data_loader = self.val_loader
            step_fn = self.val_step_fn
        else:
            data_loader = self.train_loader
            step_fn = self.train_step_fn
        if data_loader is None:
            return None
        mini_batch_losses = []
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            loss = step_fn(x_batch, y_batch)
            mini_batch_losses.append(loss)
        return np.mean(mini_batch_losses)
    def set_seed(self, seed=42):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        np.random.seed(seed)
    def train(self, n_epochs, seed=42):
        self.set_seed(seed)
        for i in range(n_epochs):
            self.total_epochs += 1
            loss = self._mini_batch()
            self.losses.append(loss)
            with torch.no_grad():
                val_loss = self._mini_batch(validation=True)
                self.val_losses.append(val_loss)
            if self.writer:
                scalars = {'training': loss}
                if val_loss is not None:
                    scalars.update({'validation': val_loss})
                self.writer.add_scalars(main_tag='loss',
                                    tag_scalar_dict=scalars,
                                    global_step=i)
        if self.writer:
            self.writer.flush()
    def save_checkpoint(self, filename):
        checkpoint = {
            'epoch':self.total_epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss':self.losses,
            'val_losses':self.val_losses
        }
        torch.save(checkpoint, filename)

    def load_checkpoint(self, filename):
        # Loads dictionary
        checkpoint = torch.load(filename)

        # Restore state for model and optimizer
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.total_epochs = checkpoint['epoch']
        self.losses = checkpoint['loss']
        self.val_losses = checkpoint['val_losses']

        self.model.train()  # always use TRAIN for resuming training

    def predict(self, x):
        # Set is to evaluation mode for predictions
        self.model.eval()
        # Takes aNumpy input and make it a float tensor
        x_tensor = torch.as_tensor(x).float()
        # Send input to device and uses model for prediction
        y_hat_tensor = self.model(x_tensor.to(self.device))
        # Set it back to train mode
        self.model.train()
        # Detaches it, brings it to CPU and back to Numpy
        return y_hat_tensor.detach().cpu().numpy()

    def plot_losses(self):
        fig = plt.figure(figsize=(10, 4))
        plt.plot(self.losses, label='Training losses', c='b')
        if self.val_losses:
            plt.plot(self.val_losses, label='Validation losses', c='r')
        plt.yscale('log')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend()
        plt.tight_layout()
        plt.show()
        return fig
    def add_graph(self):
        if self.train_loader and self.writer:
            x_dummy, y_dummy = next(iter(self.train_loader))
            self.writer.add_graph(self.model, x_dummy.to(self.device))
true_b = 1
true_w = 2
N = 100

# Data Generation
np.random.seed(42)
x = np.random.rand(N, 1)
y = true_b + true_w * x + (.1 * np.random.randn(N, 1))
#figure1(x, y)

train_loader, val_loader = prepare_data(x, y)
model, optimizer, loss_fn = modal_configuration()
#print(model.state_dict())
sbs = StepByStep(model, loss_fn, optimizer)
sbs.set_loaders(train_loader, val_loader)
sbs.set_tensorboard('classy')
#print(sbs.model == model)
#print(sbs.model)
sbs.train(n_epochs=200)
print(model.state_dict()) # remember, model == sbs.model
print(sbs.total_epochs)
#plt.plot(x, y)
#fig = sbs.plot_losses()
new_data = np.array([.5, .3, .7]).reshape(-1, 1)
predictions = sbs.predict(new_data)
print(predictions)

sbs.save_checkpoint('model_checkpoint.pth')
model, optimizer, loss_fn = modal_configuration()
print(model.state_dict())

new_sbs = StepByStep(model, loss_fn, optimizer)
new_sbs.load_checkpoint('model_checkpoint.pth')
#print(model.state_dict())

print(model.state_dict())
new_sbs.set_loaders(train_loader, val_loader)
new_sbs.train(n_epochs=50)
print(sbs.model.state_dict())
#fig = new_sbs.plot_losses()