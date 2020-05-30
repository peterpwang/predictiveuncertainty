from classes import abstractmodel, cifar100focalmodel
import torch.multiprocessing as mp

import argparse
import sys
import os

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=1, type=int, metavar='N',
                        help='number of total epochs to run (default 1)')
    parser.add_argument('--num_classes', default=100, type=int, metavar='N',
                        help='number of classes (default 100)')
    parser.add_argument('--batch_size', default=128, type=int, metavar='N',
                        help='batch size (default 128)')
    parser.add_argument('--lr', default=0.1, type=float, 
                        help='learning rate (default 0.1)')
    parser.add_argument('--focal_gamma', default=1.0, type=float, 
                        help='focal gamma (default 1.0)')
    parser.add_argument('--resume', action='store_true', 
                        help='resume from checkpoint')
    args = parser.parse_args()

    model = cifar100focalmodel.EfficientNetB7CIFAR100FocalModel(args)
    model.run()
