from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod

class AbstractImageClassificationModel(ABC):
    
    def __init__(self, epochs, num_classes, batch_size):
        self.epochs = epochs
        self.num_classes = num_classes
        self.batch_size = batch_size;

    # Load dataset and split into training and test sets.
    @abstractmethod
    def load_dataset(self):
        pass
    
    # Process data by transforming the color values into [0, 1]
    def prepare_dataset(self, train_images, train_labels, test_images, test_labels):
        train_images = train_images.astype('float32')
        train_labels = train_labels.astype('float32')
        train_images = train_images / 255.0
        test_images = test_images / 255.0

        train_labels = keras.utils.to_categorical(train_labels,self.num_classes)
        test_labels = keras.utils.to_categorical(test_labels,self.num_classes)
        return train_images, train_labels, test_images, test_labels

    # Set model
    @abstractmethod
    def define_model(self):
        pass
    
    # Display test result
    def display_results(self, history):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss=history.history['loss']
        val_loss=history.history['val_loss']

        epochs_range = range(self.epochs)

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()
        
    def run(self):

        train_images, train_labels, test_images, test_labels = self.load_dataset()

        train_images, train_labels, test_images, test_labels = self.prepare_dataset(train_images, train_labels, test_images, test_labels)

        model = self.define_model()

        history = model.fit(train_images, train_labels, 
                            epochs=self.epochs, 
                            batch_size=self.batch_size,
                            validation_data=(test_images, test_labels))

        acc = model.evaluate(test_images, test_labels, verbose=0)
        print('> %.3f' % (acc * 100.0))

        self.display_results(history)


