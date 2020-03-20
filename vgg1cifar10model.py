from classes import abstractmodel, cifar10model
import sys

if __name__ == "__main__":
    epochs = 10
    if (len(sys.argv) == 2):
        epochs = int(sys.argv[1])
    vgg1model = cifar10model.VGG1CIFAR10Model(epochs, 10, 64)
    vgg1model.run()
