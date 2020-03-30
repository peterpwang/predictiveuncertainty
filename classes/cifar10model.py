from .abstractmodel import AbstractImageClassificationModel

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F


class AbstractCIFAR10ImageClassificationModel(AbstractImageClassificationModel):

    # Load dataset and split into training and test sets.
    def load_dataset(self):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                         shuffle=False, num_workers=2)
        return trainloader, testloader


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class VGG1CIFAR10Model(AbstractCIFAR10ImageClassificationModel):
    
    # Set model
    def define_model(self):
        model = {}
        net = Net()
        net.to("cuda")
        model["net"] = net
        return model


class Resnet50CIFAR10Model(AbstractCIFAR10ImageClassificationModel):
    
    # Set model
    def define_model(self):
        model = {}
        net = torch.hub.load('pytorch/vision:v0.4.2', 'resnet50', pretrained=False)
        net.to("cuda")
        model["net"] = net
        return model

