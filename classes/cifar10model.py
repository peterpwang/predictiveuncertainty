from .abstractmodel import AbstractImageClassificationModel
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import datasets, layers, models                                           
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten


class AbstractCIFAR10ImageClassificationModel(AbstractImageClassificationModel):

    # Load dataset and split into training and test sets.
    def load_dataset(self):
        (trainX, trainY), (testX, testY) = cifar10.load_data()
        return trainX, trainY, testX, testY


class VGG1CIFAR10Model(AbstractCIFAR10ImageClassificationModel):
    
    # Set model
    def define_model(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(10, activation='softmax'))
        
        self.compile_model(model)
        return model
    

class VGG2CIFAR10Model(AbstractCIFAR10ImageClassificationModel):
    
    # Set model
    def define_model(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(10, activation='softmax'))
        
        self.compile_model(model)
        return model


class VGG3CIFAR10Model(AbstractCIFAR10ImageClassificationModel):
    
    # Set model
    def define_model(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(10, activation='softmax'))
        
        self.compile_model(model)
        return model


class VGG3DropoutCIFAR10Model(AbstractCIFAR10ImageClassificationModel):
    
    # Set model
    def define_model(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.2))
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.2))
        model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dropout(0.2))
        model.add(Dense(10, activation='softmax'))
        
        self.compile_model(model)
        return model


class VGG3DropoutDataAugmentationCIFAR10Model(VGG3DropoutCIFAR10Model):

    def run(self):

        train_images, train_labels, test_images, test_labels = self.load_dataset()

        train_images, train_labels, test_images, test_labels = self.prepare_dataset(train_images, train_labels, test_images, test_labels)

        model = self.define_model()

        # create data generator
        datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
        # prepare iterator
        it_train = datagen.flow(train_images, train_labels, batch_size=self.batch_size)
        # fit model
        steps = int(train_images.shape[0] / 64)
        history = model.fit(it_train, 
                                      steps_per_epoch=steps, 
                                      epochs=self.epochs, 
                                      validation_data=(test_images, test_labels), 
                                      verbose=0)

        _, acc = model.evaluate(test_images, test_labels, verbose=0)
        print('> %.3f' % (acc * 100.0))

        self.display_results(history)

