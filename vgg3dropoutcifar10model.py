from classes import abstractmodel, cifar10model

vgg3dropoutmodel = cifar10model.VGG3DropoutCIFAR10Model(10, 10, 64)
vgg3dropoutmodel.run()

