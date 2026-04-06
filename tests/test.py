import torch
import torch.nn.functional as F

# 输入: 3个样本，每个4维特征
x = torch.randn(3, 4)  # [batch_size, in_features]

# 权重: 输出2维，输入4维
W = torch.randn(2, 4)  # [out_features, in_features]

# 偏置: 输出2维
b = torch.randn(2)     # [out_features]

# 计算: y = x @ W.T + b
output = F.linear(x, W, b)
print(output.shape)  # [3, 2]