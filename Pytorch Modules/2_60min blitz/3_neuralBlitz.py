'''
Neural networks are constructed using the torch.nn package
A typical training procedure is as follows
* Define the neural network that has learnable parameters
* Iterate over a dataset of inputs
* Process input through the network
* Computer the loss (how wasis the output from being correct)
* Propagate gradeints back into network parameters
* Update the weights of the network typically using a simple update rule
'''
# import the required libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the net class


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation y = Wx + b
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        '''
        Forward pass of the network
        '''
        # Max pooling over a (2,2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()  # Intialize the network
print(net)
params = list(net.parameters())
print(len(params))
print(params[0].size())  # conv1's .weight

input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)

# zero the gradient buffers of all parameters and backprops with random gradients
net.zero_grad()
out.backward(torch.randn(1, 10))

output = net(input)
target = torch.randn(10)  # a dummy target for example
target = target.view(1, -1)  # make it the same shape as output
criterion = nn.MSELoss()
loss = criterion(output, target)
print(loss)


print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU

print("\n\n")
# Backprop
# To backpropagate the error all we have to do is to loss.backward()
# You need to clear the existing gradients though, else gradients will be accumulated to existing gradients
net.zero_grad()  # zeroes the gradient buffers of all parameters
print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)
loss.backward()
print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)
