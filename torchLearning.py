import torch
x = torch.arange(12, dtype=torch.float32)
print(x.shape)
X = x.reshape(3, 4)



print(X)

zeros = torch.zeros((2,3,4))
print(X[1:3])

Y = torch.exp(X)

print(X.shape, Y.shape, Y)

#Operations
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y

#concate along the axis =0 and axis = 1
X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=1)

#create boolean matrix -
Z = x == y
print(Z)


#broadcasting
a = torch.arange(3).reshape((3, 1)) # 3x1
b = torch.arange(2).reshape((1, 2)) # 1 X 2
#So the matrix will get relicated along the axis of the dimension one
print(a, b)

c = torch.arange(24).reshape((6, 4)) # 3x2
d = torch.arange(4).reshape((1, 4)) # 1 X 2
#So the matrix will get relicated along the axis of the dimension one

K = c + d
print(c, d, K)

#I think that for broadcast - the atleast one getting broadcasted shall have one dimension as one
#if the other is not having and axis as one then it will NOT be the one to get broadcasted


#Running operations can cause new memory to be allocated to host results