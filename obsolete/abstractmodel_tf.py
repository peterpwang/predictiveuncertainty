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

# Public variables
bins = 25
reliabilities_batch = []
reliabilities_epoch = []


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
            metrics=['accuracy', ece, correct_nll, incorrect_nll, correct_entropy, incorrect_entropy, tf.keras.metrics.CategoricalCrossentropy(), reliability_histogram])

    # Display test result
    def display_results(self, history):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        ece = history.history['ece']
        val_ece = history.history['val_ece']
        categorical_crossentropy = history.history['categorical_crossentropy']
        val_categorical_crossentropy = history.history['val_categorical_crossentropy']
        correct_nll = history.history['correct_nll']
        val_correct_nll = history.history['val_correct_nll']
        incorrect_nll = history.history['incorrect_nll']
        val_incorrect_nll = history.history['val_incorrect_nll']
        correct_entropy = history.history['correct_entropy']
        val_correct_entropy = history.history['val_correct_entropy']
        incorrect_entropy = history.history['incorrect_entropy']
        val_incorrect_entropy = history.history['val_incorrect_entropy']
        reliability_histogram = history.history['reliability_histogram']

        epochs_range = range(self.epochs)

        plt.figure(figsize=(16, 16))
        plt.subplot(2, 2, 1)
        plt.plot(epochs_range, correct_nll, label='Training NLL correct')
        plt.plot(epochs_range, incorrect_nll, label='Training NLL incorrect')
        plt.plot(epochs_range, loss, label='Training NLL')
        plt.plot(epochs_range, correct_entropy, label='Training Entropy correct')
        plt.plot(epochs_range, incorrect_entropy, label='Training Entropy incorrect')
        plt.legend(loc='lower right')
        plt.title('Training NLL & Entropy')

        plt.subplot(2, 2, 2)
        plt.plot(epochs_range, val_correct_nll, label='Test NLL correct')
        plt.plot(epochs_range, val_incorrect_nll, label='Test NLL incorrect')
        plt.plot(epochs_range, val_loss, label='Test NLL')
        plt.plot(epochs_range, val_correct_entropy, label='Test Entropy correct')
        plt.plot(epochs_range, val_incorrect_entropy, label='Test Entropy incorrect')
        plt.legend(loc='upper right')
        plt.title('Test NLL & Entropy')

        plt.subplot(2, 2, 3)
        plt.plot(epochs_range, loss, label='Training classfication error')
        plt.plot(epochs_range, val_loss, label='Test classfication error')
        plt.plot(epochs_range, val_ece, label='Test ECE')
        plt.legend(loc='upper right')
        plt.title('Test Error')

        plt.subplot(2, 2, 4)
        plt.plot(epochs_range, acc, label='Training accuracy')
        plt.plot(epochs_range, val_acc, label='Test accuracy')
        plt.legend(loc='upper right')
        plt.title('Training and Test accuracy')

        plt.savefig('result.png')
        
    def run(self):

        # Fix GPU error
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)

        train_images, train_labels, test_images, test_labels = self.load_dataset()

        train_images, train_labels, test_images, test_labels = self.prepare_dataset(train_images, train_labels, test_images, test_labels)

        model = self.define_model()

        history = model.fit(train_images, train_labels, 
                            epochs=self.epochs, 
                            batch_size=self.batch_size,
                            validation_data=(test_images, test_labels),
                            callbacks=[tf.keras.callbacks.BaseLogger(stateful_metrics=['reliability_histogram']), ReliabilityHistogramCallback()])

        _, acc, ece, correct_nll, incorrect_nll, correct_entropy, incorrect_entropy, crossentropy, hist = model.evaluate(test_images, test_labels, verbose=0)
        print('acc> %.3f' % acc)
        print('ece> %.3f' % ece)
        print('correct nll> %.3f' % correct_nll)
        print('correct entropy> %.3f' % correct_entropy)

        # Reliabilities histogram
        global reliabilities_epoch

        for reliability in reliabilities_epoch:
            #tf.map_fn(lambda x: print(x), reliability)
            #result = tf.keras.backend.eval(reliability)
            #print(sess.run(reliability))
            print(reliability)
            #print(np.asarray(reliability))

        self.display_results(history)


# ECE
def ece(y_true, y_pred):
    y_true = tf.math.argmax(tf.dtypes.cast(y_true, tf.int32),1)
    logits = tf.math.log(y_pred)
    return tfp.stats.expected_calibration_error(
        bins, logits=logits, labels_true=y_true)


# Correct NLL(loss)
def correct_nll(y_true, y_pred):
    index_true = tf.math.argmax(y_true, axis=1, output_type='int32')
    index_pred = tf.math.argmax(y_pred, axis=1, output_type='int32')
    correct = tf.equal(index_true, index_pred)

    y_true = tf.boolean_mask(y_true, correct)
    y_pred = tf.boolean_mask(y_pred, correct)

    return tf.math.reduce_mean(tf.keras.backend.categorical_crossentropy(y_true, y_pred))


# Incorrect NLL(loss)
def incorrect_nll(y_true, y_pred):
    index_true = tf.math.argmax(y_true, axis=1, output_type='int32')
    index_pred = tf.math.argmax(y_pred, axis=1, output_type='int32')
    incorrect = tf.not_equal(index_true, index_pred)

    y_true = tf.boolean_mask(y_true, incorrect)
    y_pred = tf.boolean_mask(y_pred, incorrect)

    return tf.math.reduce_mean(tf.keras.backend.categorical_crossentropy(y_true, y_pred))


# Correct Entropy
def correct_entropy(y_true, y_pred):
    index_true = tf.math.argmax(y_true, axis=1, output_type='int32')
    index_pred = tf.math.argmax(y_pred, axis=1, output_type='int32')
    correct = tf.equal(index_true, index_pred)

    y_true = tf.boolean_mask(y_true, correct)
    logits = tf.boolean_mask(y_pred, correct)

    return tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=logits)


# Incorrect Entropy
def incorrect_entropy(y_true, y_pred):
    index_true = tf.math.argmax(y_true, axis=1, output_type='int32')
    index_pred = tf.math.argmax(y_pred, axis=1, output_type='int32')
    incorrect = tf.not_equal(index_true, index_pred)

    y_true = tf.boolean_mask(y_true, incorrect)
    logits = tf.boolean_mask(y_pred, incorrect)

    return tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=logits)


# Reliability histogram
def reliability_histogram(y_true, y_pred):
    global reliabilities_batch
    hist_values = tf.math.reduce_sum(tf.math.multiply(y_true, y_pred), axis=1)
    value_range = [0.0, 1.0]
    hist_indexes = tf.histogram_fixed_width_bins(hist_values, value_range, nbins=bins)
    hist = tf.math.unsorted_segment_mean(hist_values, hist_indexes, bins)
    reliabilities_batch = tf.transpose(hist)

    return reliabilities_batch


# Rliability histogram callback
class ReliabilityHistogramCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        global reliabilities_batch, reliabilities_epoch
        reliabilities_epoch.append(reliabilities_batch)

