import torch


t1=torch.tensor([[1,2,3],[4,5,6]])
t2=torch.tensor([[7,8,9],[10,11,12]])

print( "t1.shape=",t1.shape, "\n", "t2.shape=",t2.shape)
result=torch.cat([t1,t2],dim=0)
print("cat dim=0", result, result.shape)

result=torch.cat([t1,t2],dim=1)
print( "cat dim=1",result, result.shape)
