import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
a = torch.tensor([2.], requires_grad=True)
y = torch.zeros((10))
gt = torch.zeros((10))

y[0] = a
y[1] = y[0] * 3
y.retain_grad()

loss = torch.sum((y-gt) ** 2)
loss.backward()
print(y.grad)
print(a.grad)