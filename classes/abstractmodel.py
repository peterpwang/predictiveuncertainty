from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.optimizers import SGD
import tensorflow_probability as tfp

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
    
    # compile model
    def compile_model(self, model):
        opt = SGD(lr=0.001, momentum=0.9)
        model.compile(optimizer=opt, 
            loss='categorical_crossentropy', 
            metrics=['accuracy', ece])

    # Display test result
    def display_results(self, history):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss=history.history['loss']
        val_loss=history.history['val_loss']

        ece=history.history['ece']
        val_ece=history.history['val_ece']

        epochs_range = range(self.epochs)

        plt.figure(figsize=(8, 8))
        plt.subplot(3, 1, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(3, 1, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')

        plt.subplot(3, 1, 3)
        plt.plot(epochs_range, ece, label='Training ECE')
        plt.plot(epochs_range, val_ece, label='Validation ECE')
        plt.legend(loc='upper right')
        plt.title('Training and Validation ECE')

        plt.savefig('result.png')
        
    def run(self):

        train_images, train_labels, test_images, test_labels = self.load_dataset()

        train_images, train_labels, test_images, test_labels = self.prepare_dataset(train_images, train_labels, test_images, test_labels)

        model = self.define_model()

        history = model.fit(train_images, train_labels, 
                            epochs=self.epochs, 
                            batch_size=self.batch_size,
                            validation_data=(test_images, test_labels))

        _, acc, ece = model.evaluate(test_images, test_labels, verbose=0)
        print('acc> %.3f' % acc)
        print('ece> %.3f' % ece)

        self.display_results(history)


# ECE
def ece(y_true, y_pred):
    y_true = tf.math.argmax(tf.dtypes.cast(y_true, tf.int32),1)
    logits = tf.math.log(y_pred)
    return tfp.stats.expected_calibration_error(
        10, logits=logits, labels_true=y_true)

