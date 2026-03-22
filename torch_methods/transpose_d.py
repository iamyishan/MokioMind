
# transpose转换维度
import torch

t1=torch.Tensor([[1,2,3],[4,5,6]]) # (2,3)

print("转换前的形状:",t1.shape)
t1=t1.transpose(0,1)
print(t1) # (3,2)
print("转换后的形状:",t1.shape)