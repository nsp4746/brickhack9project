'''
This is it, this where the all pieces begin to fall into place.
We are going to train a classifier.
We can load that data for it into a numpy array,
The data can then be converted into a torch tensor.

For images, packages such as Pillow, OpenCV exist
For audio, packages such as scipy and librosa exist
For text, either raw Python or Cython based loading, or NLTK and SpaCy are useful as well

For vision, pytorch has a package called torrch vision that has data loaders such as ImageNet, CIFAR10, MNIST,
and data transformers for images, viz., torchvision.datasets and torch.utils.data.DataLoader.
This is great because it provides a huge convenience and avoids writing boilerplate code. 

In this training of a classifer we will do use CIFAR10 dataset.

'''
#####################
'''
Step 1. Load and normailze the CIFAR10 training set
'''

# Importing the required libraries


import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # Normalizing the data
batch_size = 4

trainset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform)
# The data is downloaded in the folder ./data, time to load it
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=2)
# The data is loaded in the trainloader variable
# The data is shuffled and the batch size is 4

# It's nw time to loader the test data set
trainset = torchvision.daasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform
)
# The data is downloaded in the folder ./data, time to load it
testLoader = torch.utils.data.DataLoader(trainset,
                                         batch_size=batch_size,
                                         shuffle=False,
                                         num_workers=2)
# The data is loaded in the testLoader variable

classes = (
    'plane',
    'car',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
)
# Loaded into a tuple

# Lets make so we can see some of the training images
# Import the required libraries

# write function to show an image


def imageShow(img):
    img = img/2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(1, 2, 0))
    plt.show()


dataiter = iter(trainloader)  # get random images
images, labels = next(dataiter)

# show the images
imageShow(torchvision.utils.make_grid(images))
# print labels
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

# 2. Define a convulational neural network
# Import the required libraries


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # 3 input channels, 6 output channels, 5x5 kernel
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)  # 2x2 kernel
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all the dimensions except the batch size
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
