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

if __name__ == "__main__":

    import torch.optim as optim
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

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

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
    trainset = torchvision.datasets.CIFAR10(
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
        plt.imshow(np.transpose(npimg,(1, 2, 0)))
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
            # flatten all the dimensions except the batch size
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x


    net = Net()
    net.to(device)

    # 3. Define a loss function and optimizer
    # Using Classification Cross-Entropy loss and SGD with momentum
    # import the required librariers

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # lr is the learning rate, momentum is the momentum term

    # 4. Train the network
    # This is where the magic happens. Simply loop over the data iteraltor and feed the inputs to the network and optimizer

    for epoch in range(2):

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(f'[{epoch+1}, {i+1}] loss: {running_loss/2000:.3f}')
                running_loss = 0.0
    print("finished tracking")

    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)

    # 5. Test the network on the test data
    # We have trained the network for 2 passes over the training dataset.
    dataiter = iter(testLoader)
    images, labels = next(dataiter)
    # print images
    imageShow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

    net = Net()
    net.load_state_dict(torch.load(PATH))
    net.to(device)

    outputs = net(images)
    _, predicted = torch.max(outputs, 1)
    print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(batch_size)))

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testLoader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the network on the 10000 test images: {100*correct/total}%')

    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    # again with no gradients needed
    with torch.no_grad():
        for data in testLoader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1
    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy  = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class {classname:5s} is {accuracy :.1f}%')
# Classifer is complete