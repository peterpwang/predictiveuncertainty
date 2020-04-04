from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, RunningAverage
from ignite.contrib.metrics import GpuInfo
from ignite.contrib.handlers.tqdm_logger import ProgressBar

# Helper libraries
import logging
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

from .custommetrics import *

# Public variables
bins = 25
device = "cuda"


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
    def compile_model(self, net):
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        criterion = nn.CrossEntropyLoss().cuda()
        return optimizer, criterion
        #print("device:", next(net.parameters()).device)

    # Display test result
    def display_results(self, history):
        val_loss = history['val_loss']
        test_loss = history['test_loss']
        val_acc = history['val_accuracy']
        test_acc = history['test_accuracy']
        test_ece = history['test_ece']
        val_nll = history['val_nll']
        test_nll = history['test_nll']
        val_correct_nll = history['val_correct_nll']
        test_correct_nll = history['test_correct_nll']
        val_incorrect_nll = history['val_incorrect_nll']
        test_incorrect_nll = history['test_incorrect_nll']
        val_correct_entropy = history['val_correct_entropy']
        test_correct_entropy = history['test_correct_entropy']
        val_incorrect_entropy = history['val_incorrect_entropy']
        test_incorrect_entropy = history['test_incorrect_entropy']

        epochs_range = range(self.epochs)

        plt.figure(figsize=(16, 16))
        plt.subplot(2, 2, 1)
        plt.plot(epochs_range, val_correct_nll, label='Train NLL correct')
        plt.plot(epochs_range, val_incorrect_nll, label='Train NLL incorrect')
        plt.plot(epochs_range, val_nll, label='Train NLL')
        plt.plot(epochs_range, val_correct_entropy, label='Train Entropy correct')
        plt.plot(epochs_range, val_incorrect_entropy, label='Train Entropy incorrect')
        plt.legend(loc='lower right')
        plt.title('Train NLL & Entropy')

        plt.subplot(2, 2, 2)
        plt.plot(epochs_range, test_correct_nll, label='Test NLL correct')
        plt.plot(epochs_range, test_incorrect_nll, label='Test NLL incorrect')
        plt.plot(epochs_range, test_nll, label='Test NLL')
        plt.plot(epochs_range, test_correct_entropy, label='Test Entropy correct')
        plt.plot(epochs_range, test_incorrect_entropy, label='Test Entropy incorrect')
        plt.legend(loc='upper right')
        plt.title('Test NLL & Entropy')

        plt.subplot(2, 2, 3)
        plt.plot(epochs_range, val_loss, label='Train classfication error')
        plt.plot(epochs_range, test_loss, label='Test classfication error')
        plt.plot(epochs_range, test_ece, label='Test ECE')
        plt.legend(loc='upper right')
        plt.title('Test Error')

        plt.subplot(2, 2, 4)
        plt.plot(epochs_range, val_acc, label='Train accuracy')
        plt.plot(epochs_range, test_acc, label='Test accuracy')
        plt.legend(loc='upper right')
        plt.title('Train and Test accuracy')

        plt.savefig('result.png')
        plt.close()

        # Reliability plot
        bin_boundaries = np.linspace(0, 1, bins)

        test_accuracy_sum_bins = history['test_accuracy_sum_bins']
        test_accuracy_num_bins = history['test_accuracy_num_bins']

        for i in range(self.epochs):
            plt.figure(figsize=(8, 16))
            plt.subplot(2, 1, 1)
            plt.plot(bin_boundaries, bin_boundaries)
            plt.bar(bin_boundaries, test_accuracy_sum_bins[i], width=0.015, label='Test Accuracy')
            plt.legend(loc='upper right')
            plt.title('Reliability Plot')
    
            plt.subplot(2, 1, 2)
            plt.bar(bin_boundaries, test_accuracy_num_bins[i], width=0.015, label='Test Samples')
            plt.legend(loc='upper right')
            plt.title('Reliability Plot')

            plt.savefig('reliability_plot_' + '{:03d}'.format(i+1) + '.png')
            
            if i%19 == 0:
                plt.close()
        

    def run(self):

        FORMAT = '%(asctime)-15s %(message)s'
        logging.basicConfig(level=logging.INFO, format=FORMAT)


        # GPU related settings
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Torch:", torch.__version__, "(CPU+GPU)" if  torch.cuda.is_available() else "(CPU)")

        # Load dataset
        train_loader, validation_loader, test_loader = self.load_dataset()

        # Create model
        net = self.define_model()
        optimizer, criterion = self.compile_model(net)

        metrics = {
            'accuracy': Accuracy(output_transform=lambda output: (torch.round(output[0]), output[1]), device=device),
            'loss': Loss(criterion, device=device),
            'nll': NLL(device=device),
            'correct_nll': CorrectNLL(device=device),
            'incorrect_nll': IncorrectNLL(device=device),
            'correct_entropy': CorrectCrossEntropy(device=device),
            'incorrect_entropy': IncorrectCrossEntropy(device=device),
            'ece': ECE(device=device)
        }

        trainer = create_supervised_trainer(net, optimizer, criterion, device=device, non_blocking=True)
        train_evaluator = create_supervised_evaluator(net, metrics=metrics, device=device, non_blocking=True)
        validation_evaluator = create_supervised_evaluator(net, metrics=metrics, device=device, non_blocking=True)
        test_evaluator = create_supervised_evaluator(net, metrics=metrics, device=device, non_blocking=True)

        # track train & validation accuracy and loss at end of each epoch
        history = { "loss": [], "val_loss": [], "test_loss": [], 
                "accuracy": [], "val_accuracy": [], "test_accuracy": [], 
                "nll": [], "val_nll": [], "test_nll": [], 
                "correct_nll": [], "val_correct_nll": [], "test_correct_nll": [], 
                "incorrect_nll": [], "val_incorrect_nll": [], "test_incorrect_nll": [], 
                "correct_entropy": [], "val_correct_entropy": [], "test_correct_entropy": [], 
                "incorrect_entropy": [], "val_incorrect_entropy": [], "test_incorrect_entropy": [],
                "test_ece": [], 
                "test_accuracy_sum_bins": [], 
                "test_accuracy_num_bins": []}

        def log_train_results(trainer):
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
            print("Train Results - Epoch: {:3d}  Accuracy: {:.3f} Loss: {:.3f} Entropy: {:.3f} {:.3f}  "
                  .format(trainer.state.epoch, accuracy, loss, correct_entropy, incorrect_entropy), end=" ")

        def log_validation_results(trainer):
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
            print("Validation Results - Accuracy: {:.3f} Loss: {:.3f} Entropy: {:.3f} {:.3f} NLL {:.3f} {:.3f}"
                  .format(accuracy, loss, correct_entropy, incorrect_entropy, correct_nll, incorrect_nll))
    
        def log_test_results(trainer):
            test_evaluator.run(test_loader)
            metrics = test_evaluator.state.metrics
            accuracy = metrics['accuracy']
            loss = metrics['loss']
            nll = metrics['nll']
            correct_nll = metrics['correct_nll']
            incorrect_nll = metrics['incorrect_nll']
            correct_entropy = metrics['correct_entropy']
            incorrect_entropy = metrics['incorrect_entropy']
            ece, accuracy_sum_bins, accuracy_num_bins = metrics['ece']
            history['test_accuracy'].append(accuracy)
            history['test_loss'].append(loss)
            history['test_nll'].append(nll)
            history['test_correct_nll'].append(correct_nll)
            history['test_incorrect_nll'].append(incorrect_nll)
            history['test_correct_entropy'].append(correct_entropy)
            history['test_incorrect_entropy'].append(incorrect_entropy)
            history['test_ece'].append(ece)
            print("Test Results - Accuracy: {:.3f} Loss: {:.3f} Entropy: {:.3f} {:.3f} NLL {:.3f} {:.3f}"
                  .format(accuracy, loss, correct_entropy, incorrect_entropy, correct_nll, incorrect_nll))

            # Reliability plot
            history['test_accuracy_sum_bins'].append(accuracy_sum_bins)
            history['test_accuracy_num_bins'].append(accuracy_num_bins)
    
        #trainer.add_event_handler(Events.EPOCH_COMPLETED, log_train_results)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, log_validation_results)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, log_test_results)

        GpuInfo().attach(trainer, name='gpu')
        pbar = ProgressBar()
        pbar.attach(trainer, metric_names=['gpu:0 mem(%)', 'gpu:0 util(%)'])

        # Track loss during epoch and print out in progress bar
        #RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')

        # kick off training...
        trainer.run(train_loader, max_epochs=self.epochs)

        self.display_results(history)
