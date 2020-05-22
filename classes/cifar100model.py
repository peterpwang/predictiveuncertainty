from .abstractmodel import AbstractImageClassificationModel

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import geffnet


# Abstract Cifar 100 model
class AbstractCIFAR100ImageClassificationModel(AbstractImageClassificationModel):

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


# EfficientNet B0 Cifar 100 model
class EfficientNetB0CIFAR100Model(AbstractCIFAR100ImageClassificationModel):
    
    # Set model
    def define_model(self):
        return geffnet.efficientnet_b0(pretrained=False, drop_rate=0.25, drop_connect_rate=0.2).cuda()


# EfficientNet B2 Cifar 100 model
class EfficientNetB2CIFAR100Model(AbstractCIFAR100ImageClassificationModel):
    
    # Set model
    def define_model(self):
        return geffnet.efficientnet_b2(pretrained=False, drop_rate=0.25, drop_connect_rate=0.2).cuda()


# EfficientNet B5 Cifar 100 model
class EfficientNetB5CIFAR100Model(AbstractCIFAR100ImageClassificationModel):
    
    # Set model
    def define_model(self):
        return geffnet.efficientnet_b5(pretrained=False, drop_rate=0.25, drop_connect_rate=0.2).cuda()


# EfficientNet B7 Cifar 100 model
class EfficientNetB7CIFAR100Model(AbstractCIFAR100ImageClassificationModel):
    
    # Set model
    def define_model(self):
        return geffnet.efficientnet_b7(pretrained=False, drop_rate=0.25, drop_connect_rate=0.2).cuda()

