from classes import abstractmodel, cifar10model
import sys

if __name__ == "__main__":
    epochs = 10
    if (len(sys.argv) == 2):
        epochs = int(sys.argv[1])
    vgg3dropoutmodel = cifar10model.VGG3DropoutCIFAR10Model(epochs, 10, 64)
    vgg3dropoutmodel.run()
