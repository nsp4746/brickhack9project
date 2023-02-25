'''
What are Tensors?
Tensors are a data structure similar to arrays and matrices,
This is what we use in pytorch 
'''
import torch
import numpy as np

# Intialize a tensor from direct data
data = [[1,2],[3,4],[5,6],[7,8]]
x_data = torch.tensor(data)

# Intialize a tensor from a numpy array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# Intialize from another tensor
x_ones = torch.ones_like(x_data)
print(f"Ones Tensor: {x_ones}")
'''
Output:
Ones Tensor: tensor([[1, 1],
        [1, 1],
        [1, 1],
        [1, 1]])
'''
x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the fact we have int in data
print(f"Random Tensor: {x_rand}")
'''
!!!!!!!!!!!!!!! Random everytime !!!!!!!!!!!!!!!!!
Output:
Random Tensor: tensor([[0.0154, 0.0248],
        [0.4973, 0.6055],
        [0.7298, 0.9158],
        [0.4129, 0.3288]])
'''

# Create a tensor with random or constant values

shape = (3,3,) # generates 3x3 tensor
rand_tensor = torch.rand(shape) # random values
ones_tensor = torch.ones(shape) # ones value
zeros_tensor = torch.zeros(shape) # zeros value

print(f"Random Tensor:\n{rand_tensor}\n")
print(f"Ones Tensor:\n{ones_tensor}\n")
print(f"Zeros Tensor:\n{zeros_tensor}\n")

# Tensor Attributes
tensor = torch.rand(6,6) # 6,6 value
print(f"Shape of tensor: {tensor.shape}")  # will print 6,6
print(f"Datatype of tensor: {tensor.dtype}") # will print float32
print(f"Device tensor is stored on: {tensor.device}") # will print cpu, as i do not have a gpu on my laptop

if torch.cuda.is_available(): # moves the tensor to the GPU if available, will differently have to clone this on my pc
    tensor = tensor.to('cuda')
    print(f"Device tensor is stored on: {tensor.device}") # will print cuda

# Tensors can also be sliced, like strings, and numpy arrays

tensor = torch.ones(2,2)
tensor[:,1] = 0
print(tensor)

# Tensors can also be concatenated

catTensor = torch.cat([tensor, tensor, tensor], dim=1) # will concatenate the tensor 3 times, along the 1st axis
print(catTensor)

# Multiplaying tensors
# Create a tensor that has random float values
tensor = torch.rand(2,2)
print(f"\ntensor:\n{tensor}\n")

# This computes the element-wise prduct
print(f"tensor.matmul(tensor.T) \n {tensor.mul(tensor)}\n")
# Alternative syntax:
print(f"tensor * tensor \n {tensor * tensor}\n")

# Matrix multimplication
print(f"tensor.matmul(tensor.T) \n {tensor.matmul(tensor.T)}\n")
# Alternative syntax:
print(f"tensor @ tensor.T \n {tensor @ tensor.T}\n")

# of course, you can also add values

tensor = torch.ones(2,2)
print(f"tensor:\n{tensor}\n")
tensor.add_(5)
print(f"tensor add (5):\n{tensor}\n")



