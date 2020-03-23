from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.optimizers import SGD
import tensorflow_probability as tfp
import tensorflow.keras.backend as K

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
            #metrics=['accuracy', ece, tf.keras.metrics.CategoricalCrossentropy(), correct_nll])
            metrics=['accuracy', ece, correct_nll])

    # Display test result
    def display_results(self, history):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        ece = history.history['ece']
        val_ece = history.history['val_ece']
        #categorical_crossentropy = history.history['categorical_crossentropy']
        #val_categorical_crossentropy = history.history['val_categorical_crossentropy']
        correct_nll = history.history['correct_nll']
        val_correct_nll = history.history['val_correct_nll']
        incorrect_nll = history.history['incorrect_nll']
        val_incorrect_nll = history.history['val_incorrect_nll']

        epochs_range = range(self.epochs)

        plt.figure(figsize=(12, 12))
        plt.subplot(2, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(2, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')

        plt.subplot(2, 2, 3)
        plt.plot(epochs_range, ece, label='Training ECE')
        plt.plot(epochs_range, val_ece, label='Validation ECE')
        plt.legend(loc='upper right')
        plt.title('Training and Validation ECE')

        plt.subplot(2, 2, 4)
        plt.plot(epochs_range, correct_nll, label='Training Correct NLL')
        plt.plot(epochs_range, val_correct_nll, label='Validation Correct NLL')
        plt.plot(epochs_range, incorrect_nll, label='Training Incorrect NLL')
        plt.plot(epochs_range, val_incorrect_nll, label='Validation Incorrect NLL')
        plt.legend(loc='upper right')
        plt.title('Training and Validation NLL')

        plt.savefig('result.png')
        
    def run(self):

        #cfg = K.tf.ConfigProto(gpu_options={'allow_growth': True})
        #K.set_session(K.tf.Session(config=cfg))

        train_images, train_labels, test_images, test_labels = self.load_dataset()

        train_images, train_labels, test_images, test_labels = self.prepare_dataset(train_images, train_labels, test_images, test_labels)

        model = self.define_model()

        history = model.fit(train_images, train_labels, 
                            epochs=self.epochs, 
                            batch_size=self.batch_size,
                            validation_data=(test_images, test_labels))

        _, acc, ece, correct_nll, incorrect_nll = model.evaluate(test_images, test_labels, verbose=0)
        print('acc> %.3f' % acc)
        print('ece> %.3f' % ece)
        print('correct nll> %.3f' % correct_nll)
        print('incorrect nll> %.3f' % incorrect_nll)

        self.display_results(history)


# ECE
def ece(y_true, y_pred):
    y_true = tf.math.argmax(tf.dtypes.cast(y_true, tf.int32),1)
    logits = tf.math.log(y_pred)
    return tfp.stats.expected_calibration_error(
        25, logits=logits, labels_true=y_true)


# Correct NLL(loss)
def correct_nll(y_true, y_pred):
    index_true = tf.math.argmax(y_true, axis=1, output_type='int32')
    index_pred = tf.math.argmax(y_pred, axis=1, output_type='int32')
    correct = tf.cast(tf.equal(index_true, index_pred), tf.float32)
    y_true = correct * y_true
    y_pred = correct * y_pred
    return tf.math.reduce_sum(tf.keras.backend.categorical_crossentropy(y_true, y_pred))


# Incorrect NLL(loss)
def incorrect_nll(y_true, y_pred):
    index_true = tf.math.argmax(y_true, axis=1, output_type='int32')
    index_pred = tf.math.argmax(y_pred, axis=1, output_type='int32')
    incorrect = tf.cast(tf.not_equal(index_true, index_pred), tf.float32)
    y_true = incorrect * y_true
    y_pred = incorrect * y_pred
    return tf.math.reduce_sum(tf.keras.backend.categorical_crossentropy(y_true, y_pred))
