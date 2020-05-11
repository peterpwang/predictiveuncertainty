import os
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets.vision import VisionDataset
from PIL import Image


def make_mini_imagenet_dataset(directory):
    instances = []
    directory = os.path.expanduser(directory)
    y_file = open(directory + "/y.bin", "rb")
    y = y_file.read(1)
    idx = 0
    while y:
        item = idx*96*96*3, y
        instances.append(item)
        y = y_file.read(1)
        idx = idx + 1

    y_file.close()

    return instances


class MiniImagenetDatasetFolder(VisionDataset):
    """A generic data loader where the samples are arranged in this way: ::
        X.bin, Y.bin
        https://www.cs.waikato.ac.nz/~ml/sins10/

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.

     Attributes:
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root, transforms=None, transform=None, target_transform=None):
        super(MiniImagenetDatasetFolder, self).__init__(root, transforms, transform, target_transform)

        samples = make_mini_imagenet_dataset(root)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 samples in subfolders of: " + self.root + "\n"))

        self.samples = samples
        self.targets = [s[1] for s in samples]

        self.x_file = open(root + "/X.bin", "rb")

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        offset, target = self.samples[index]
        self.x_file.seek(offset)
        sample = self.x_file.read(96*96*3)
        sample = Image.frombytes('RGB', (96,96), sample)
        if self.transform is not None:
            sample = self.transform(sample)

        target = int.from_bytes(target, byteorder='big')
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


    def __len__(self):
        return len(self.samples)
