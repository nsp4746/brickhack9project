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
import torch
import torchvision
import torchvision.transforms as transforms


transform = transforms.Compose([transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]) # Normalizing the data
batch_size = 4 

trainset = torchvision.datasets.CIFAR10(
    root = './data', 
    train = True, 
    download = True, 
    transform = transform)
# The data is downloaded in the folder ./data, time to load it
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size = batch_size,
                                          shuffle = True,
                                          num_workers = 2)
# The data is loaded in the trainloader variable
# The data is shuffled and the batch size is 4

# It's nw time to loader the test data set