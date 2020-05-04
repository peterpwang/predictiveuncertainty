from .abstractmodel import AbstractImageClassificationModel

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet


# Abstract ImageNet model
class AbstractImageNetImageClassificationModel(AbstractImageClassificationModel):

    # Load dataset and split into training and test sets.
    def load_dataset(self):
        train_transform = transforms.Compose([
            #transforms.Resize(256),
            #transforms.CenterCrop(224),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        dataset = torchvision.datasets.ImageNet(root='./data', train=True,
                                        download=True, transform=train_transform)
        trainset, validationset = torch.utils.data.random_split(dataset, [45000, 5000])
        trainloader = torch.utils.data.DataLoader(trainset, 
                batch_size=128,
                shuffle=True, 
                num_workers=2, 
                pin_memory=True)
        validationloader = torch.utils.data.DataLoader(validationset, 
                batch_size=128,
                shuffle=True, 
                num_workers=2, 
                pin_memory=True)

        testset = torchvision.datasets.ImageNet(root='./data', train=False,
                                       download=True, transform=test_transform)
        testloader = torch.utils.data.DataLoader(testset, 
                batch_size=128,
                shuffle=False, 
                num_workers=2, 
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


# Test ImageNet model
class TestImageNetModel(AbstractImageNetImageClassificationModel):

    # Set model
    def define_model(self):
        return TestNet().cuda()


# Resnet50 ImageNet model
class Resnet50ImageNetModel(AbstractImageNetImageClassificationModel):
    
    # Set model
    def define_model(self):
        return torch.hub.load('pytorch/vision:v0.4.2', 'resnet50', pretrained=False).cuda()


# Densenet121 ImageNet model
class Densenet121ImageNetModel(AbstractImageNetImageClassificationModel):
    
    # Set model
    def define_model(self):
        return torch.hub.load('pytorch/vision:v0.4.2', 'densenet121', pretrained=False).cuda()


# EfficientNet B0 ImageNet model
class EfficientNetB0ImageNetModel(AbstractImageNetImageClassificationModel):
    
    # Set model
    def define_model(self):
        return EfficientNet.from_name('efficientnet-b0').cuda()
