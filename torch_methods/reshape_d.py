import torch
x=torch.arange(1,7)
print(x)

y=torch.reshape(x,(2,3 ))
print(y)

#使用-1推断
z=torch.reshape(x,(3,-1,))
print(z)
