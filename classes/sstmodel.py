from .abstractmodel import AbstractClassificationModel

import os
import random
import spacy

import torch
from torchtext import data    
import torchtext
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ignite.engine import create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, RunningAverage

from .sstmetrics import *

from .thirdparty.TreeLSTMSentiment import Constants
# NEURAL NETWORK MODULES/LAYERS
from .thirdparty.TreeLSTMSentiment.model import TreeLSTMSentiment
# DATA HANDLING CLASSES
from .thirdparty.TreeLSTMSentiment.tree import Tree
from .thirdparty.TreeLSTMSentiment.vocab import Vocab
# DATASET CLASS FOR SICK DATASET
from .thirdparty.TreeLSTMSentiment.dataset import SSTDataset
# METRICS CLASS FOR EVALUATION
from .thirdparty.TreeLSTMSentiment.metrics import Metrics
# UTILITY FUNCTIONS
from .thirdparty.TreeLSTMSentiment.utils import load_word_vectors, build_vocab
# CONFIG PARSER
from .thirdparty.TreeLSTMSentiment.config import parse_args
# TRAIN AND TEST HELPER FUNCTIONS
from .thirdparty.TreeLSTMSentiment.trainer import SentimentTrainer


SEED = 1234
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


# Abstract SST model
class AbstractSSTTextClassificationModel(AbstractClassificationModel):

    def __init__(self, args):
        super(AbstractSSTTextClassificationModel, self).__init__(args)

        self.model_name = "dependency"
        self.input_dim = 300
        self.mem_dim = 168
        self.num_classes = 3 # 0 1 2 (1 neutral)
        self.data_path = "data/sst/sst/"
        
    def define_model(self):
        pass

    def load_dataset(self):
        pass


    def create_criterion(self):
        criterion = nn.NLLLoss().cuda(0)
        return criterion
        

    def create_trainer(self, net, embedding_model, criterion, optimizer, device):
        global args
        args = parse_args(type=1)
        args.fine_grain = False
        args.cuda = "cuda"
        args.batchsize = 64
        args.elmlr = 0.1

        trainer = SentimentTrainer(args, net, embedding_model, criterion, optimizer)
        return trainer
    

    def run(self):

        # GPU related settings
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Torch:", torch.__version__, "(CPU+GPU)" if  torch.cuda.is_available() else "(CPU)")
        torch.cuda.set_device(0)

        # Create directories
        if not os.path.isdir('checkpoints'):
            os.mkdir('checkpoints')
        if not os.path.isdir('results'):
            os.mkdir('results')

        # Load dataset
        train_dir = os.path.join(self.data_path, 'train/')
        dev_dir = os.path.join(self.data_path, 'dev/')
        test_dir = os.path.join(self.data_path, 'test/')

        # write unique words from all token files
        token_files = [os.path.join(split, 'sents.toks') for split in [train_dir, dev_dir, test_dir]]
        vocab_file = os.path.join(self.data_path,'vocab-cased.txt') # use vocab-cased
        # get vocab object from vocab file previously written
        vocab = Vocab(filename=vocab_file)
        print('==> SST vocabulary size : %d ' % vocab.size())

        # train
        train_file = os.path.join(self.data_path,'sst_train.pth')
        if os.path.isfile(train_file):
            train_dataset = torch.load(train_file)
        else:
            train_dataset = SSTDataset(train_dir, vocab, self.num_classes, False, self.model_name)
            torch.save(train_dataset, train_file)

        # dev
        dev_file = os.path.join(self.data_path,'sst_dev.pth')
        if os.path.isfile(dev_file):
            dev_dataset = torch.load(dev_file)
        else:
            dev_dataset = SSTDataset(dev_dir, vocab, self.num_classes, False, self.model_name)
            torch.save(dev_dataset, dev_file)

        # test
        test_file = os.path.join(self.data_path,'sst_test.pth')
        if os.path.isfile(test_file):
            test_dataset = torch.load(test_file)
        else:
            test_dataset = SSTDataset(test_dir, vocab, self.num_classes, False, self.model_name)
            torch.save(test_dataset, test_file)

        # Create model
        criterion = self.create_criterion()

        model = TreeLSTMSentiment(
                device, vocab.size(),
                self.input_dim, self.mem_dim,
                self.num_classes, self.model_name, criterion
        ).cuda(0)

        embedding_model = nn.Embedding(vocab.size(), self.input_dim).cuda(0)

        #define trainer
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=self.learning_rate, weight_decay=5e-4)

        trainer = self.create_trainer(model, embedding_model, criterion, optimizer, device)

        emb_file = os.path.join(args.data, 'sst_embed.pth')
        if os.path.isfile(emb_file):
            emb = torch.load(emb_file).cuda()
        else:
            # load glove embeddings and vocab
            glove_vocab, glove_emb = load_word_vectors(os.path.join('data/sst/glove/','glove.840B.300d'))
            print('==> GLOVE vocabulary size: %d ' % glove_vocab.size())

            emb = torch.zeros(vocab.size(),glove_emb.size(1))

            for word in vocab.labelToIdx.keys():
                if glove_vocab.getIndex(word):
                    emb[vocab.getIndex(word)] = glove_emb[glove_vocab.getIndex(word)]
                else:
                    emb[vocab.getIndex(word)] = torch.Tensor(emb[vocab.getIndex(word)].size()).normal_(-0.05,0.05)
            torch.save(emb, emb_file)
            print('done creating emb, quit')

        embedding_model.state_dict()['weight'].copy_(emb)

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

        # Create metrics
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
    
        # kick off training...
        for epoch in range(args.epochs):
            train_loss = trainer.train(train_dataset)
            train_loss, train_pred = trainer.test(train_dataset)
            dev_loss, dev_pred = trainer.test(dev_dataset)
            test_loss, test_pred = trainer.test(test_dataset)

            train_acc = calculate_accuracy(train_pred, train_dataset.labels)
            dev_acc = calculate_accuracy(dev_pred, dev_dataset.labels)
            test_acc = calculate_accuracy(test_pred, test_dataset.labels)
            print('==> Train loss   : %f \t' % train_loss, end="")
            print('Epoch ', epoch, 'train percentage ', train_acc)
            print('Epoch ', epoch, 'dev percentage ', dev_acc)
            print('Epoch ', epoch, 'test percentage ', test_acc)
            train_nll = calculate_nll(train_pred, train_dataset.labels.int())
            dev_nll = calculate_nll(dev_pred, dev_dataset.labels.int())
            test_nll = calculate_nll(test_pred, test_dataset.labels.int())
            print('Epoch ', epoch, 'train NLL ', train_nll)
            print('Epoch ', epoch, 'dev NLL ', dev_nll)
            print('Epoch ', epoch, 'test NLL ', test_nll)

            #log_train_results(trainer)
            #log_test_results(trainer);


        self.output_results(history)


# Tree LSTM model
class TreeLSTMNet(AbstractSSTTextClassificationModel):
    pass    

