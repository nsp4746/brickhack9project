'''
Torch autogradient is a pytorch's automatic
differentiation engine that powers neural network training
in this tutorial, a basic understanding of how autogradient helps a neural network train'''

# This will only work on a CPU, no GPUs

import torch
from torchvision.models import resnet18, ResNet18_Weights
model = resnet18(weights=ResNet18_Weights.DEFAULT)
data = torch.rand(1, 3, 64, 64)
labels = torch.rand(1, 1000)
# intialize the model, data, labels

# forward pass, means running through the input data through the model through each of its layers to make a prediction, this is called the forward pass
prediction = model(data)

loss = (prediction - labels).sum()
loss.backward()  # backward pass

# This is loading the optimzer, with a learning rate of 0.01 and a momentum 0.9
optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
optim.step()  # gradient descent

'''
Differention in AutoGrad
create two tensors a and b with requires gradient. this tells the autograd that every operation on them should be tracked
'''
a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)
# Create a new tensor, Q, from a and b
# Q = 3a^3 - b^2
Q = 3*a**3 - b**2
external_grad = torch.tensor([1, 1])
# Backpropagate Q with respect to a and b
Q.backward(gradient=external_grad)
# check if the gradients are correct
print(9*a**2 == a.grad)
print(-2*b == b.grad)
