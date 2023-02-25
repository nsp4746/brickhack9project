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
#import the required libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the net class
class Net(nn.module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel 
        self.conv1 = nn.Conv2d(1,6,5)
        self.conv2 = nn.Conv2d(6,16,5)
        # an affine operation y = Wx + b
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        