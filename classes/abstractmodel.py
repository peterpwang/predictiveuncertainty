from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, RunningAverage

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

from .custommetrics import *

# Public variables
bins = 25

class AbstractImageClassificationModel(ABC):
    
    def __init__(self, epochs, num_classes, batch_size):
        self.epochs = epochs
        self.num_classes = num_classes
        self.batch_size = batch_size;

    # Load dataset and split into training and test sets.
    @abstractmethod
    def load_dataset(self):
        pass
    
    # Set model
    @abstractmethod
    def define_model(self):
        pass
    
    # compile model
    def compile_model(self, model):
        net = model["net"]
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        model["criterion"] = criterion
        model["optimizer"] = optimizer

    # Display test result
    def display_results(self, history):
        loss = history['loss']
        val_loss = history['val_loss']
        acc = history['accuracy']
        val_acc = history['val_accuracy']
        val_ece = history['val_ece']
        #categorical_crossentropy = history['categorical_crossentropy']
        #val_categorical_crossentropy = history['val_categorical_crossentropy']
        nll = history['nll']
        val_nll = history['val_nll']
        correct_nll = history['correct_nll']
        val_correct_nll = history['val_correct_nll']
        incorrect_nll = history['incorrect_nll']
        val_incorrect_nll = history['val_incorrect_nll']
        correct_entropy = history['correct_entropy']
        val_correct_entropy = history['val_correct_entropy']
        incorrect_entropy = history['incorrect_entropy']
        val_incorrect_entropy = history['val_incorrect_entropy']

        epochs_range = range(self.epochs)

        plt.figure(figsize=(16, 16))
        plt.subplot(2, 2, 1)
        plt.plot(epochs_range, correct_nll, label='Training NLL correct')
        plt.plot(epochs_range, incorrect_nll, label='Training NLL incorrect')
        plt.plot(epochs_range, nll, label='Training NLL')
        plt.plot(epochs_range, correct_entropy, label='Training Entropy correct')
        plt.plot(epochs_range, incorrect_entropy, label='Training Entropy incorrect')
        plt.legend(loc='lower right')
        plt.title('Training NLL & Entropy')

        plt.subplot(2, 2, 2)
        plt.plot(epochs_range, val_correct_nll, label='Test NLL correct')
        plt.plot(epochs_range, val_incorrect_nll, label='Test NLL incorrect')
        plt.plot(epochs_range, val_nll, label='Test NLL')
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

        # Reliability plot
        bin_boundaries = np.linspace(0, 1, bins)

        val_accuracy_sum_bins = history['val_accuracy_sum_bins']
        val_accuracy_num_bins = history['val_accuracy_num_bins']

        for i in range(self.epochs):
            plt.figure(figsize=(8, 16))
            plt.subplot(2, 1, 1)
            plt.plot(bin_boundaries, bin_boundaries)
            plt.bar(val_accuracy_sum_bins[i], val_accuracy_sum_bins[i], width=0.02, label='Accuracy')
            plt.legend(loc='upper right')
            plt.title('Reliability Plot')
    
            plt.subplot(2, 1, 2)
            plt.bar(bin_boundaries, val_accuracy_num_bins[i], width=0.02, label='Accuracy')
            plt.legend(loc='upper right')
            plt.title('Reliability Plot')

            plt.savefig('reliability_plot_' + str(i+1) + '.png')
        

    def run(self):

        # GPU related settings
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Torch:", torch.__version__, "(CPU+GPU)" if  torch.cuda.is_available() else "(CPU)")

        # Load dataset
        train_loader, validation_loader = self.load_dataset()

        # Create model
        model = self.define_model()
        net = model["net"]

        self.compile_model(model)
        optimizer = model["optimizer"]
        criterion = model["criterion"]

        trainer = create_supervised_trainer(net, optimizer, criterion, device=device)

        metrics = {
            'accuracy': Accuracy(output_transform=lambda output: (torch.round(output[0]), output[1])),
            'loss': Loss(criterion),
            'nll': NLL(),
            'correct_nll': CorrectNLL(),
            'incorrect_nll': IncorrectNLL(),
            'correct_entropy': CorrectCrossEntropy(),
            'incorrect_entropy': IncorrectCrossEntropy(),
            'ece': ECE()
        }
        train_evaluator = create_supervised_evaluator(net, metrics=metrics, device=device)
        validation_evaluator = create_supervised_evaluator(net, metrics=metrics, device=device)

        # track train & validation accuracy and loss at end of each epock
        history = { "loss": [], "val_loss": [], 
                "accuracy": [], "val_accuracy": [], 
                "nll": [], "val_nll": [], 
                "correct_nll": [], "val_correct_nll": [], 
                "incorrect_nll": [], "val_incorrect_nll": [], 
                "correct_entropy": [], "val_correct_entropy": [], 
                "incorrect_entropy": [], "val_incorrect_entropy": [],
                "val_ece": [], 
                "val_accuracy_sum_bins": [], 
                "val_accuracy_num_bins": []}

        def log_validation_results(trainer):
            train_evaluator.run(train_loader)
            metrics = train_evaluator.state.metrics
            accuracy = metrics['accuracy']
            loss = metrics['loss']
            nll = metrics['nll']
            correct_nll = metrics['correct_nll']
            incorrect_nll = metrics['incorrect_nll']
            correct_entropy = metrics['correct_entropy']
            incorrect_entropy = metrics['incorrect_entropy']
            history['accuracy'].append(accuracy)
            history['loss'].append(loss)
            history['nll'].append(nll)
            history['correct_nll'].append(correct_nll)
            history['incorrect_nll'].append(incorrect_nll)
            history['correct_entropy'].append(correct_entropy)
            history['incorrect_entropy'].append(incorrect_entropy)
            print("Train Results - Epoch: {:3d}  Accuracy: {:.4f} Loss: {:.4f} NLL: {:.4f}"
                  .format(trainer.state.epoch, accuracy, loss, nll), end=" ")

            validation_evaluator.run(validation_loader)
            metrics = validation_evaluator.state.metrics
            accuracy = metrics['accuracy']
            loss = metrics['loss']
            nll = metrics['nll']
            correct_nll = metrics['correct_nll']
            incorrect_nll = metrics['incorrect_nll']
            correct_entropy = metrics['correct_entropy']
            incorrect_entropy = metrics['incorrect_entropy']
            ece, accuracy_sum_bins, accuracy_num_bins = metrics['ece']
            history['val_accuracy'].append(accuracy)
            history['val_loss'].append(loss)
            history['val_nll'].append(nll)
            history['val_correct_nll'].append(correct_nll)
            history['val_incorrect_nll'].append(incorrect_nll)
            history['val_correct_entropy'].append(correct_entropy)
            history['val_incorrect_entropy'].append(incorrect_entropy)
            history['val_ece'].append(ece)
            print("Validation Results - Accuracy: {:.4f} Loss: {:.4f} NLL: {:.4f}"
                  .format(accuracy, loss, nll))

            # Reliability plot
            history['val_accuracy_sum_bins'].append(accuracy_sum_bins)
            history['val_accuracy_num_bins'].append(accuracy_num_bins)
    
        trainer.add_event_handler(Events.EPOCH_COMPLETED, log_validation_results)

        # Track loss during epoch and print out in progress bar
        #RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')

        # kick off training...
        trainer.run(train_loader, max_epochs=self.epochs)

        self.display_results(history)
