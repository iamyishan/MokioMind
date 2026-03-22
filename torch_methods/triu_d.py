import torch

x=torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
print(torch.triu(x))

#输出：
# tensor([[1, 2, 3],
#         [0, 5, 6],
#         [0, 0, 9]])

print(torch.triu(x,diagonal=1))
#输出：
# tensor([[0, 2, 3],
#         [0, 0, 6],
#         [0, 0, 0]])

print(torch.triu(x,diagonal=-1))

#输出：
# tensor([[1, 2, 3],
#         [4, 5, 6],
#         [0, 8, 9]])