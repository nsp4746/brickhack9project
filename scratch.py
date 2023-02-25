import torch
import numpy as np

data = [[1,2], [3,4]]
x_data = torch.tensor(data)
tensor = torch.rand(3,4)
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")
if torch.cuda.is_available():
    tensor = tensor.to('cuda')

tensor = torch.ones(4,4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:,0]}")
print(f"Last column: {tensor[...,-1]}")
tensor[:,1] = 0
print(tensor)

t1 = torch.cat([tensor,tensor,tensor], dim=1)
print(t1)

y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T,out=y3)

z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
torch.mul(tensor,tensor,out=z3)
print(f"{tensor}\n")
tensor.add_(5)
print(tensor)

agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))
