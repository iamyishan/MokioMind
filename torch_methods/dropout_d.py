import torch
import torch.nn as nn

# Dropout 在训练时会随机将一部分神经元的输出置为 0，同时缩放剩余神经元的值以保持期望不变。
dropout_layer=nn.Dropout(p=0.5)
t1=torch.Tensor([1,2,3])
t2=dropout_layer(t1)
print(t2)