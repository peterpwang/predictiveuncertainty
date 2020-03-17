from classes import abstractmodel, cifar10model

vgg3ddamodel = cifar10model.VGG3DropoutDataAugmentationCIFAR10Model(10, 10, 64)
vgg3ddamodel.run()

