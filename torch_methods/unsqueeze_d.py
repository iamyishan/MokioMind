import torch

t1=torch.tensor([1,2,3])
t2=t1.unsqueeze(0)

print(t1.shape)
print(t2)
print(t2.shape)