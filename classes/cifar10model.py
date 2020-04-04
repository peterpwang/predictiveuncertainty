from .abstractmodel import AbstractImageClassificationModel

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F


# Abstract Cifar 10 model
class AbstractCIFAR10ImageClassificationModel(AbstractImageClassificationModel):

    # Load dataset and split into training and test sets.
    def load_dataset(self):
        transform = transforms.Compose([
            # This two lines help to improve accuracy but take about 240s/epoch on ResNet50.
            # Without them, one epoch only takes about 30s on ResNet50.
            #transforms.Resize(256),
            #transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
        trainset, validationset = torch.utils.data.random_split(dataset, [45000, 5000])

        trainloader = torch.utils.data.DataLoader(trainset, 
                batch_size=64,
                shuffle=True, 
                num_workers=0, 
                pin_memory=True)
        validationloader = torch.utils.data.DataLoader(validationset, 
                batch_size=64,
                shuffle=True, 
                num_workers=0, 
                pin_memory=True)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, 
                batch_size=64,
                shuffle=False, 
                num_workers=0, 
                pin_memory=True)
        #print(len(trainloader), len(validationloader), len(testloader))

        return trainloader, validationloader, testloader


# Test model, quick and simple
class TestNet(nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()
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


# Test Cifar 10 model
class TestCIFAR10Model(AbstractCIFAR10ImageClassificationModel):

    # Set model
    def define_model(self):
        return TestNet().cuda()


# Resnet50 Cifar model
class Resnet50CIFAR10Model(AbstractCIFAR10ImageClassificationModel):
    
    # Set model
    def define_model(self):
        return torch.hub.load('pytorch/vision:v0.4.2', 'resnet50', pretrained=True).cuda()

