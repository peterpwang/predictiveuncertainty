from classes import abstractmodel, cifar10model
import sys

if __name__ == "__main__":
    epochs = 10
    if (len(sys.argv) == 2):
        epochs = int(sys.argv[1])
    model = cifar10model.Resnet50CIFAR10Model(epochs, 10, 64)
    model.run()