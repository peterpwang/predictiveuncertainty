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
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

from .custommetrics import *

# Public variables
bins = 25
device = "cuda"


class AbstractImageClassificationModel(ABC):
    
    def __init__(self, args):
        self.epochs = args.epochs
        self.num_classes = args.num_classes
        self.batch_size = args.batch_size
        self.learning_rate = args.lr
        self.focal_gamma = args.focal_gamma
        self.resume = args.resume

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
        print("Learning rate is set to ", self.learning_rate)
        optimizer = optim.SGD(net.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss().cuda(0)
        return optimizer, criterion

    # Display test result
    def output_results(self, history):
        loss = history['loss']
        test_loss = history['test_loss']
        acc = history['accuracy']
        test_acc = history['test_accuracy']
        test_ece = history['test_ece']
        nll = history['nll']
        test_nll = history['test_nll']
        correct_nll = history['correct_nll']
        test_correct_nll = history['test_correct_nll']
        incorrect_nll = history['incorrect_nll']
        test_incorrect_nll = history['test_incorrect_nll']
        correct_entropy = history['correct_entropy']
        test_correct_entropy = history['test_correct_entropy']
        incorrect_entropy = history['incorrect_entropy']
        test_incorrect_entropy = history['test_incorrect_entropy']

        epochs_range = range(len(loss))

        # output to plot
        plt.figure(figsize=(16, 16))
        plt.subplot(2, 2, 1)
        plt.plot(epochs_range, correct_nll, label='Train NLL correct')
        plt.plot(epochs_range, incorrect_nll, label='Train NLL incorrect')
        plt.plot(epochs_range, nll, label='Train NLL')
        plt.plot(epochs_range, correct_entropy, label='Train Entropy correct')
        plt.plot(epochs_range, incorrect_entropy, label='Train Entropy incorrect')
        plt.legend(loc='upper right')
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
        plt.plot(epochs_range, loss, label='Train classfication error')
        plt.plot(epochs_range, test_loss, label='Test classfication error')
        plt.plot(epochs_range, test_ece, label='Test ECE')
        plt.legend(loc='upper right')
        plt.title('Test Error')

        plt.subplot(2, 2, 4)
        plt.plot(epochs_range, acc, label='Train accuracy')
        plt.plot(epochs_range, test_acc, label='Test accuracy')
        plt.legend(loc='lower right')
        plt.title('Train and Test accuracy')

        plt.savefig('results/' + type(self).__name__ + '_result.png')
        plt.close()

        # Output to tsv
        tsv_file = open('results/' + type(self).__name__ + '_result.tsv', 'w+')
        tsv_file.write('Train NLL,Train NLL Correct,Train NLL Incorrect,'
                + 'Train CE Correct,Train CE Incorrect,'
                + 'Test NLL,Test NLL Correct,Test NLL Incorrect,'
                + 'Test CE Correct,Test CE Incorrect,'
                + 'Train Error,Test Error,Test ECE,'
                + 'Train Accuracy,Test Accuracy\n')
        for i in range(len(loss)):
            tsv_file.write('{:6.2f}'.format(nll[i]) + ',' + '{:6.2f}'.format(correct_nll[i]) + ',' + '{:6.2f}'.format(incorrect_nll[i]) + ',')
            tsv_file.write('{:6.2f}'.format(correct_entropy[i]) + ',' + '{:6.2f}'.format(incorrect_entropy[i]) + ',') 
            tsv_file.write('{:6.2f}'.format(test_nll[i]) + ',' + '{:6.2f}'.format(test_correct_nll[i]) + ',' + '{:6.2f}'.format(test_incorrect_nll[i]) + ',')
            tsv_file.write('{:6.2f}'.format(test_correct_entropy[i]) + ',' + '{:6.2f}'.format(test_incorrect_entropy[i]) +',') 
            tsv_file.write('{:6.2f}'.format(loss[i]) + ',' + '{:6.2f}'.format(test_loss[i]) + ',' + '{:6.2f}'.format(test_ece[i]) + ',') 
            tsv_file.write('{:6.2f}'.format(acc[i]) + ',' + '{:6.2f}'.format(test_acc[i]) + '\n') 
        tsv_file.close()

        # Reliability plot
        bin_boundaries = np.linspace(0, 1, bins)

        test_accuracy_sum_bins = history['test_accuracy_sum_bins']
        test_accuracy_num_bins = history['test_accuracy_num_bins']

        for i in range(len(loss)):
            plt.figure(figsize=(8, 16))
            plt.subplot(2, 1, 1)
            plt.plot(bin_boundaries, bin_boundaries)
            plt.bar(bin_boundaries, test_accuracy_sum_bins[i], width=0.015, label='Test Accuracy')
            plt.legend(loc='upper left')
            plt.title('Reliability Plot')
    
            # Calculate sample %
            count_samples = 0.0
            for j in range(len(test_accuracy_num_bins[i])):
                count_samples += test_accuracy_num_bins[i][j]

            plt.subplot(2, 1, 2)
            plt.plot(bin_boundaries, bin_boundaries)
            plt.bar(bin_boundaries, test_accuracy_num_bins[i]/count_samples, width=0.015, label='Test Sample Percentage')
            plt.legend(loc='upper left')
            plt.title('Reliability Plot')

            plt.savefig('results/' + type(self).__name__ + '_reliability_plot_' + '{:03d}'.format(i+1) + '.png')
            
            plt.close()
        

    def save_checkpoint(self, trainer, net, optimizer, start_epoch, history):
        state = {
            'net': net.state_dict(),
            'optimizer': optimizer,
            'start_epoch': trainer.state.epoch,
            'history': history
        }
        torch.save(state, './checkpoints/checkpoint.bin')
        #print("Checkpoint saved.")
        

    def load_checkpoint(self, net, optimizer, history):
        checkpoint = torch.load('./checkpoints/checkpoint.bin')
        net.load_state_dict(checkpoint['net'])
        optimizer = checkpoint['optimizer']
        start_epoch = checkpoint['start_epoch']
        history = checkpoint['history']
        print("State reloaded.", start_epoch)
        return net, optimizer, start_epoch, history
        

    def run(self):

        #FORMAT = '%(asctime)-15s %(message)s'
        #logging.basicConfig(level=logging.INFO, format=FORMAT)

        # GPU related settings
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Torch:", torch.__version__, "(CPU+GPU)" if  torch.cuda.is_available() else "(CPU)")
        torch.cuda.set_device(0)

        # Create directories
        if not os.path.isdir('checkpoints'):
            os.mkdir('checkpoints')
        if not os.path.isdir('results'):
            os.mkdir('results')

        # Create model
        net = self.define_model()
        optimizer, criterion = self.compile_model(net)

        # Track train & test accuracy and loss at end of each epoch
        history = { "loss": [], "test_loss": [], 
                "accuracy": [], "test_accuracy": [], 
                "nll": [], "test_nll": [], 
                "correct_nll": [], "test_correct_nll": [], 
                "incorrect_nll": [], "test_incorrect_nll": [], 
                "correct_entropy": [], "test_correct_entropy": [], 
                "incorrect_entropy": [], "test_incorrect_entropy": [],
                "test_ece": [], 
                "test_accuracy_sum_bins": [], 
                "test_accuracy_num_bins": []}

        # Resume training
        start_epoch = 0
        if self.resume:
            net, optimizer, start_epoch, history = self.load_checkpoint(net, optimizer, history)
            optimizer, criterion = self.compile_model(net)

        # Create trainer
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

        def setup_state(trainer):
            trainer.state.epoch = start_epoch

        def save_state(trainer):
            self.save_checkpoint(trainer, net, optimizer, start_epoch, history)

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
            ece, accuracy_sum_bins, accuracy_num_bins = metrics['ece']
            history['accuracy'].append(accuracy)
            history['loss'].append(loss)
            history['nll'].append(nll)
            history['correct_nll'].append(correct_nll)
            history['incorrect_nll'].append(incorrect_nll)
            history['correct_entropy'].append(correct_entropy)
            history['incorrect_entropy'].append(incorrect_entropy)
            print("Epoch[{}] Train Results - Accuracy: {:.3f} Loss: {:.3f} Entropy: {:.3f} {:.3f} NLL {:.3f} {:.3f}"
                  .format(trainer.state.epoch, accuracy, loss, correct_entropy, incorrect_entropy, correct_nll, incorrect_nll), end=" ")
    
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
    
        trainer.add_event_handler(Events.STARTED, setup_state)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, log_train_results)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, log_test_results)
        trainer.add_event_handler(Events.EPOCH_COMPLETED(every=5), save_state)

        #GpuInfo().attach(trainer, name='gpu')
        #pbar = ProgressBar()
        #pbar.attach(trainer, metric_names=['gpu:0 mem(%)', 'gpu:0 util(%)'])

        # Track loss during epoch and print out in progress bar
        #RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')

        # Load dataset
        train_loader, validation_loader, test_loader = self.load_dataset()

        # kick off training...
        trainer.run(train_loader, max_epochs=self.epochs + start_epoch)

        self.output_results(history)

