import torch
import torch.nn as nn

layer=nn.Linear(in_features=3,out_features=4,bias=True)
t1=torch.Tensor([1,2,3])
out=layer(t1)
print(out.shape)

t2=torch.Tensor([[1,2,3]])

# 这里应用的w和b是随机的，真实训练里会在optimizer上更新
out2=layer(t2)
print(out2.shape)