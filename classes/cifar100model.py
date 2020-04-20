from .abstractmodel import AbstractImageClassificationModel

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F


# Abstract Cifar 100 model
class AbstractCIFAR100ImageClassificationModel(AbstractImageClassificationModel):

    # Load dataset and split into training and test sets.
    def load_dataset(self):
        train_transform = transforms.Compose([
            # These two lines help to improve accuracy but take about 240s/epoch on ResNet50.
            # Without them, one epoch only takes about 30s on ResNet50.
            # But later test with these lines got good results (~20s). So let's put them here.
            #transforms.Resize(256),
            #transforms.CenterCrop(224),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        dataset = torchvision.datasets.CIFAR100(root='./data', train=True,
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

        testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                       download=True, transform=test_transform)
        testloader = torch.utils.data.DataLoader(testset, 
                batch_size=128,
                shuffle=False, 
                num_workers=2, 
                pin_memory=True)
        #print(len(trainloader), len(validationloader), len(testloader))

        return trainloader, validationloader, testloader


# Resnet50 Cifar 100 model
class Resnet50CIFAR100Model(AbstractCIFAR100ImageClassificationModel):
    
    # Set model
    def define_model(self):
        return torch.hub.load('pytorch/vision:v0.4.2', 'resnet50', pretrained=False).cuda()


# Densenet121 Cifar 100 model
class Densenet121CIFAR100Model(AbstractCIFAR100ImageClassificationModel):
    
    # Set model
    def define_model(self):
        return torch.hub.load('pytorch/vision:v0.4.2', 'densenet121', pretrained=False).cuda()

