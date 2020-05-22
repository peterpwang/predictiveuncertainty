from .abstractmodel import AbstractImageClassificationModel
from .util import MiniImagenetDatasetFolder, make_mini_imagenet_dataset

import os
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets.vision import VisionDataset
from efficientnet_pytorch import EfficientNet
from PIL import Image
import geffnet


# Abstract Mini Imagenet model
class AbstractMiniImagenetImageClassificationModel(AbstractImageClassificationModel):

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

        dataset = MiniImagenetDatasetFolder(root='./data/mini-imagenet/', transform=train_transform)
        medianset, testset = torch.utils.data.random_split(dataset, [90000, 10000])
        trainset, validationset = torch.utils.data.random_split(medianset, [80000, 10000])
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
        testloader = torch.utils.data.DataLoader(testset, 
                batch_size=128,
                shuffle=False, 
                num_workers=2, 
                pin_memory=True)

        return trainloader, validationloader, testloader


# Resnet50 Mini Imagenet model
class Resnet50MiniImagenetModel(AbstractMiniImagenetImageClassificationModel):
    
    # Set model
    def define_model(self):
        return torch.hub.load('pytorch/vision:v0.4.2', 'resnet50', pretrained=False).cuda()


# Densenet121 Mini Imagenet model
class Densenet121MiniImagenetModel(AbstractMiniImagenetImageClassificationModel):
    
    # Set model
    def define_model(self):
        return torch.hub.load('pytorch/vision:v0.4.2', 'densenet121', pretrained=False).cuda()


# EfficientNet B0 Mini Imagenet model
class EfficientNetB0MiniImagenetModel(AbstractMiniImagenetImageClassificationModel):
    
    # Set model
    def define_model(self):
        #return EfficientNet.from_name('efficientnet-b0').cuda()
        return geffnet.efficientnet_b0(pretrained=False, drop_rate=0.25, drop_connect_rate=0.2).cuda()


# EfficientNet B2 Mini Imagenet model
class EfficientNetB2MiniImagenetModel(AbstractMiniImagenetImageClassificationModel):
    
    # Set model
    def define_model(self):
        #return EfficientNet.from_name('efficientnet-b2').cuda()
        return geffnet.efficientnet_b2(pretrained=False, drop_rate=0.25, drop_connect_rate=0.2).cuda()


# EfficientNet B5 Mini Imagenet model
class EfficientNetB5MiniImagenetModel(AbstractMiniImagenetImageClassificationModel):
    
    # Set model
    def define_model(self):
        return EfficientNet.from_name('efficientnet-b5').cuda()


# EfficientNet B7 Mini Imagenet model
class EfficientNetB7MiniImagenetModel(AbstractMiniImagenetImageClassificationModel):
    
    # Set model
    def define_model(self):
        return EfficientNet.from_name('efficientnet-b7').cuda()


# EfficientNet B8 Mini Imagenet model
class EfficientNetB8MiniImagenetModel(AbstractMiniImagenetImageClassificationModel):
    
    # Set model
    def define_model(self):
        return EfficientNet.from_name('efficientnet-b8').cuda()

