import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import torch.nn as nn

"""
We generate synthetic data here, get the output labels based on a predefined w and b
We are trying to see how the gradient descend works
"""
true_b = 1
true_w = 2
N = 100
np.random.seed(42) # 42nd random
x = np.random.rand(N, 1) # this will be creating a ndarray with 100 rows and 1 column and the values
                         # are [0, 1)
epsilon = .1 * np.random.randn(N, 1) #noise of the same dimensions as the x input ndarray, mean=0, variance=1
y = epsilon + x * true_w + true_b
print(x)
print(y)
print(epsilon)

"""
Now we will split the artificial data into training and validation sets, 
training will have 80% and validation 20% of the given data set that is artificial in this case 
"""

idx = np.arange(N)
np.random.shuffle(idx)
train_idx = idx[: int(N * .8)]
validate_idx = idx[int(N * .8):]
x_train, y_train = x[train_idx], y[train_idx]
x_validation, y_validation = x[validate_idx], y[validate_idx]

"""
Random initialization of the w and b, in this case we already know the real w and b but
we are trying to device a method to get to the optimal values starting with a guess value of the w and b
"""
np.random.seed(42)
b = np.random.randn(1)
w = np.random.randn(1)

print(b,w)


"""
Now, lets calculate the predicted yhat using the above w and b 
"""
yhat = w * x_train + b

"""
Batch descend - when all the samples are used to compute the loss
Mini Batch is the loss for a 1 to N
Stochastic is the loss for one point
"""

error = yhat - y_train
loss = (error**2).mean();
print(loss)

"""
We can try and find the losses for all the possible values of w and b
Lets find the losses for a range of Ws and Bs
First find the ranges--> Find linear space
"""

b_range = np.linspace(true_b - 3, true_b + 3, 101)
w_range = np.linspace(true_w - 3, true_w + 3, 101)
print(b_range)
print(w_range)

"""
Now we want to find the grid i.e. the combinations of w_range and b_range
"""
bs, ws = np.meshgrid(b_range, w_range)
print(bs.shape)
print(ws.shape)
#bs)
#print(ws)

# test on the axis selection
#x_test = np.array([[1],[2]])
#print(x_test)
#print(x_test.sum(axis=0))
#print(x_test.sum(axis=1))
dummy_x = x_train[0]
#print(dummy_x)
dummy_yhat = bs + ws * dummy_x
#print(dummy_yhat)
#print(ws)
#print(bs)
def myfunc(x):
    print(x)
    return x * ws + bs
all_predictions = np.apply_along_axis(
    func1d=myfunc,
    axis=1,
    arr=x_train #[[1],[2]]
)

all_labels = y_train.reshape(-1,1,1)
#print(all_labels.shape)

all_errors = all_predictions - all_labels

all_losses = (all_errors**2).mean() #MSE loss

#Compute gradient for a given w and b
b_grad = 2 * error.mean()
w_grad = 2 * (x_train * error).mean()

lr = .1
print(w, b)

w = w - lr * w_grad
b = b - lr * b_grad
print(w, b)

#Standardization scaling the train, validate and test to have 0 mean and unit vairance
scalar = StandardScaler(with_std=True, with_mean=True)
scalar.fit(x_train) # fit shall be done on the train set only as to not have the test and validaiton set info bleed into the train mean and sd
scalar.transform(x_train)
scalar.transform(x_validation)


#implementation of the gradient decent in np
epoch = 1000
eta = .1
for i in range(epoch):
    yhat = w * x_train + b
    error = yhat - y_train
    loss = (error**2).mean()
    w_grad = 2 * (error * x_train).mean()
    b_grad = 2 * error.mean()
    w = w - w_grad * eta
    b = b - b_grad * eta

print('W and B')
print(w, b)


# Sanity the same results using the Sci-kit
linr = LinearRegression()
linr.fit(x_train, y_train)
print(linr.intercept_, linr.coef_[0])


# Welcome to Tensors
scalar = torch.tensor(3.14159)
vector = torch.tensor([1, 2, 3])
matrix = torch.ones((2, 3), dtype=torch.float)
tensor = torch.randn((2, 3, 4), dtype=torch.float)

#reshape vs view, they are both used to reshape the given tensor but the view makes sure its a copy of the original tensor
same_matrix = matrix.view(1, 6)
#Same matrix is the same tensor and so will change if the matrix rensor is changed
print(same_matrix)

different_matrix = matrix.new_tensor(matrix.view(1, 6))
print('different_matrix')
print(different_matrix)
#New tensor has an alternative sourceTensor.clone.detach() as it is neater so recommended
another_matrix = matrix.view(1, 6).clone().detach()
another_matrix[0, 1] = 5
print(another_matrix, matrix)

#Convert np array into tensor, remember the torch.tensor creates the another one like copy
#for referential we can use the as_tensor,
dummy_list = np.array([1,2,3])
dummy_tensor = torch.as_tensor(dummy_list)
dummy_tensor[0] = 100
dummy_list[2] = 201
print(dummy_tensor, dummy_list)

#We can also get numpy array back
print(dummy_tensor.numpy())

#The code shall always have the option being GPU ready and torch has ways to achieve

device = 'cuda' if torch.cuda.is_available() else 'cpu' # this is the python Ternery operator or Conditonal statement
print(device)


gpu_tensor = torch.as_tensor(x_train).to(device)
print(gpu_tensor[0])

x_train_tensor = torch.as_tensor(x_train).float().to(device)
y_train_tensor = torch.as_tensor(y_train).float().to(device)
#to convert a GPU list back to numoy
#print(x_train_tensor.cpu().float().numpy())

#TO use torch for gradient computation - see that the w and b require the gradient but not the x_train
#The tensor for a learnable paramater reqires gradient
torch.manual_seed(42)
b = torch.randn(1, dtype=torch.float).to(device)
b.requires_grad_()

w = torch.randn(1, dtype=float).to(device)
w.requires_grad_()

print(b, w.size())#the approcah to have the device and gradient is just fine but we need to check another approach

lr = .1
torch.manual_seed(42)
#There is another approch to get device as well as the gradient property
b = torch.randn(1, dtype=torch.float, requires_grad=True, device=device)
w = torch.randn(1, dtype=torch.float, requires_grad=True, device=device)
#Auto Grad in Action
n_epochs = 1000
for i in range(n_epochs):
  yhat = b + w * x_train_tensor
  error = yhat - y_train_tensor
  loss = (error ** 2).mean()
  #b_grad = 2 * (error).mean()
  #w_grad = 2 * (x_train_tensor * error).mean()
  loss.backward()
  with torch.no_grad():
      b.add_(-1 * lr * b.grad) #b_grad removed by the torch's .grad #Or the other in place operation like : b -= lr * b_grad would work but the gradient upgrade shall be inplace
      w.add_(-1 * lr * w.grad) #w_grad removed by the torch's .grad #Or the other in place operation like : b -= lr * b_grad would work but the gradient upgrade shall be inplace
  b.grad.zero_()
  w.grad.zero_()
print(b, w)
#Now, the loss function is interested in the b, w, yhat, error but since
# x_train_tensor and y_train_tensor are not gradient applicable so are a dont care for the backward function

print(x_train_tensor.requires_grad, y_train_tensor.requires_grad)
print(loss.requires_grad, yhat.requires_grad, error.requires_grad, b.requires_grad, w.requires_grad, loss.size(), yhat.size())

#using optimizer to update and zero action
#And, using a loss function that pyTorch provides
lr = .1
torch.manual_seed(42)
b = torch.randn(1,dtype=torch.float, device=device, requires_grad=True)
a = torch.randn(1,dtype=torch.float, device=device, requires_grad=True)
n_epochs = 1000
optimizer = torch.optim.SGD([b, w], lr=lr)
loss_fn = torch.nn.MSELoss(reduction='mean')
for epoch in range(n_epochs):
    yhat = b + w * x_train_tensor
    error = yhat - y_train_tensor
    #loss = (error ** 2).mean()
    loss = loss_fn(y_train_tensor, yhat)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
print(b, w, loss.item(), loss.tolist())

#Model - pytorch has a call nn.Module that every Modal needs to inherit from
class ManualLinerRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.b = nn.Parameter(torch.randn(1, device=device, requires_grad=True, dtype=torch.float))#tell pytorch that they are the parameters of this model
        self.w = nn.Parameter(torch.randn(1, device=device, requires_grad=True, dtype=torch.float))

    def forward(self, x):
        return self.b + self.w * x

torch.manual_seed(42)
dummy = ManualLinerRegression().to(device)
print(dummy.parameters())
for param in dummy.parameters():
    print(type(param), param.size(), param)

#Model's state dic contains the param that have the learnable params
print(dummy.state_dict())
#Optimizer too has the state_dict() that can be used to checkpointing a model

#Lets use the model to get the learning

torch.manual_seed(42)
lr = .1
model = ManualLinerRegression()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
loss_fn = nn.MSELoss(reduction='mean')
n_epochs = 1000
for epoch in range(n_epochs):
    model.train()
    yhat = model(x_train_tensor)
    #error = yhat - y_train_tensor
    loss = loss_fn(yhat, y_train_tensor)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

for param in model.parameters():
    print(param)

#Nested models
linear = nn.Linear(1, 1)
print(linear.state_dict())

#We can use this model as an attribute

class MyManualLinerRegerssion(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
    def forward(self, x):
        return self.linear(x)

torch.manual_seed(42)
mymodel = MyManualLinerRegerssion()
print(mymodel.state_dict())

#Sequential models
torch.manual_seed(42)
model = nn.Sequential(nn.Linear(3,5), nn.Linear(5, 1)).to(device)
print(model.state_dict())#the parameter names are a sequence as the attributes are not named

torch.manual_seed(42)
model = nn.Sequential()
model.add_module('layer1', nn.Linear(3, 5))
model.add_module('layer2', nn.Linear(5, 1))
model.to(device)
print(model.state_dict())#the parameter names are now having names
