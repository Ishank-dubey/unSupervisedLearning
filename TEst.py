import torch
import sys
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
a = torch.tensor([2.], requires_grad=True)
y = torch.zeros((10))
gt = torch.zeros((10))

y[0] = a
y[1] = y[0] * 4
y.retain_grad()

loss = torch.sum((y-gt) ** 2)
loss.backward()
print(y.grad)
print(a.grad)

#19th November
#Lists concepts
mylist = ["banana", "cherry", "apple"]
print(mylist)
list1 = list()
print(list1)
print(mylist[1])
for i in mylist:
    print(i)

if "banana" in mylist:
    print("yes")
else:
    print("no")

print(len(mylist))
mylist.append('lemon')
print(mylist)

mylist.insert(1, "blackberry")
print(mylist)

lastitem = mylist.pop()
print(lastitem)
#reverse method is in place so if we want to not have the effect on the original list use the method
# sorted
new_list = sorted(mylist)
print(new_list)
print(mylist)

#
newlistwithelements = [0] * 5
print(newlistwithelements)

appendedList = [1,2,3,4,5] + newlistwithelements
print(appendedList)

#slicing
a = appendedList[1:5]# omit start index or the end index to default to the begining or the end of the list
print(a)
#while slicing we also give the step size [::stepSize]

#copy operations
appendedListNew = appendedList
print(appendedListNew == appendedList)

#Expression while iteration
last = [1,2,3,4,5,6]
lastsq = [x * x for x in last]
print(lastsq)

#Tupules - Immutable unlike the list - use a comma when only one element in the tupule
tupule = ("Max", )# tupule([])
print(type(tupule))
item = tupule[0]#index can be negative
print(item)
#Tupule is immutable, iteration is same for in loop as in the case of a list, in operator is also as in the list
print(item.count("Max"))#.index(element)
#convert to list and vice versa - list(tupule), tupule(list), slicing is similar to the list

test = "Max", 90, "Mars"
name, age, location = test  # number of elements must match or use a name, *i1, i3, i1 is all the elements in between
print(name, age, location)

print(sys.getsizeof((1,2,3)))#uses less as its immutable
print(sys.getsizeof([1,2,3]))# uses more bytes


#DICTONARIES
mydict = {"name":"Max", "age":98, "location":"delhi"}
print(mydict)#dict(name="Max")
del mydict["name"] #mydict.pop("name"), popitem()- last item
print(mydict)


try:
    print(mydict["name"])
except:
    print("Exception")
for key in mydict:
    print(mydict[key])
#update dict using update to merge or override
# key could be any type use carefully like tupule can also be used but cant use a list as a key as its mutable



#SETS