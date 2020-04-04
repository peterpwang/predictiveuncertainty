from classes import abstractmodel, cifar10model
import torch.multiprocessing as mp

import argparse
import sys
import os

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=1, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--num_classes', default=10, type=int, metavar='N',
                        help='number of classes')
    parser.add_argument('--batch_size', default=128, type=int, metavar='N',
                        help='batch size')
    args = parser.parse_args()

    model = cifar10model.Resnet50CIFAR10Model(args)
    model.run()
