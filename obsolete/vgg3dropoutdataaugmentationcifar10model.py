from classes import abstractmodel, cifar10model
import sys

if __name__ == "__main__":
    epochs = 10
    if (len(sys.argv) == 2):
        epochs = int(sys.argv[1])
    vgg3ddamodel = cifar10model.VGG3DropoutDataAugmentationCIFAR10Model(epochs, 10, 64)
    vgg3ddamodel.run()
