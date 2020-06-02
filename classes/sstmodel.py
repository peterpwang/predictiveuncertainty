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
        self.input_dim= 300
        self.mem_dim = 168
        self.num_classes = 3 # 0 1 2 (1 neutral)
        self.data_path = "data/sst/sst/"
        
        self.vocab = None
        self.embedding_model = None
        self.optimizer = None
        self.criterion = None
    
    # compile model
    def compile_model(self, net):
        print("Learning rate is set to ", self.learning_rate)
        #define optimizer and loss
        optimizer = optim.Adam(net.parameters())
        criterion = nn.NLLLoss().cuda(0)

        self.optimizer = optimizer
        self.criterion = criterion

        return optimizer, criterion

    # Load dataset and split into training and test sets.
    def load_dataset(self):

        train_dir = os.path.join(self.data_path, 'train/')
        dev_dir = os.path.join(self.data_path, 'dev/')
        test_dir = os.path.join(self.data_path, 'test/')

        # write unique words from all token files
        token_files = [os.path.join(split, 'sents.toks') for split in [train_dir, dev_dir, test_dir]]
        vocab_file = os.path.join(self.data_path,'vocab-cased.txt') # use vocab-cased
        # get vocab object from vocab file previously written
        self.vocab = Vocab(filename=vocab_file)
        print('==> SST vocabulary size : %d ' % self.vocab.size())

        # train
        train_file = os.path.join(self.data_path,'sst_train.pth')
        if os.path.isfile(train_file):
            train_dataset = torch.load(train_file)
        else:
            train_dataset = SSTDataset(train_dir, self.vocab, self.num_classes, False, self.model_name)
            torch.save(train_dataset, train_file)

        # dev
        dev_file = os.path.join(self.data_path,'sst_dev.pth')
        if os.path.isfile(dev_file):
            dev_dataset = torch.load(dev_file)
        else:
            dev_dataset = SSTDataset(dev_dir, self.vocab, self.num_classes, False, self.model_name)
            torch.save(dev_dataset, dev_file)

        # test
        test_file = os.path.join(self.data_path,'sst_test.pth')
        if os.path.isfile(test_file):
            test_dataset = torch.load(test_file)
        else:
            test_dataset = SSTDataset(test_dir, self.vocab, self.num_classes, False, self.model_name)
            torch.save(test_dataset, test_file)

        #set batch size
        BATCH_SIZE = 64

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        #Load an iterator
        train_iterator = data.BucketIterator(train_dataset, 
            batch_size = BATCH_SIZE,
            sort_key = lambda x: len(x.text),
            sort_within_batch=True,
            device = device)
        validation_iterator = data.BucketIterator(dev_dataset, 
            batch_size = BATCH_SIZE,
            sort_key = lambda x: len(x.text),
            sort_within_batch=True,
            device = device)
        test_iterator = data.BucketIterator(test_dataset, 
            batch_size = BATCH_SIZE,
            sort_key = lambda x: len(x.text),
            sort_within_batch=True,
            device = device)

        return train_iterator, validation_iterator, test_iterator


    def create_trainer(self, net, optimizer, criterion, metrics, device):
        global args
        args = parse_args(type=1)
        args.fine_grain = False
        args.cuda = "cuda"
        args.batchsize = 64
        args.elmlr = 0.1

        trainer = SentimentTrainer(args, net, self.embedding_model ,criterion, optimizer)
        train_evaluator = create_supervised_evaluator(net, metrics=metrics, device=device, non_blocking=True, prepare_batch=self.prepare_batch)
        validation_evaluator = create_supervised_evaluator(net, metrics=metrics, device=device, non_blocking=True, prepare_batch=self.prepare_batch)
        test_evaluator = create_supervised_evaluator(net, metrics=metrics, device=device, non_blocking=True, prepare_batch=self.prepare_batch)

        return trainer, train_evaluator, validation_evaluator, test_evaluator


# Tree LSTM model
class TreeLSTMNet(AbstractSSTTextClassificationModel):
    
    # Set model
    def define_model(self):

        # initialize model
        model = TreeLSTMSentiment(
                "cuda", self.vocab.size(),
                self.input_dim, self.mem_dim,
                self.num_classes, self.model_name, self.criterion
            )

        self.embedding_model = nn.Embedding(self.vocab.size(), self.input_dim).cuda()

        # for words common to dataset vocab and GLOVE, use GLOVE vectors
        # for other words in dataset vocab, use random normal vectors
        emb_file = os.path.join(self.data_path, 'sst_embed.pth')
        if os.path.isfile(emb_file):
            emb = torch.load(emb_file)
        else:

            # load glove embeddings and vocab
            glove_vocab, glove_emb = load_word_vectors(os.path.join("data/sst/glove",'glove.840B.300d'))
            print('==> GLOVE vocabulary size: %d ' % glove_vocab.size())

            emb = torch.zeros(self.vocab.size(),glove_emb.size(1))

            for word in self.vocab.labelToIdx.keys():
                if glove_vocab.getIndex(word):
                    emb[self.vocab.getIndex(word)] = glove_emb[glove_vocab.getIndex(word)]
                else:
                    emb[self.vocab.getIndex(word)] = torch.Tensor(emb[self.vocab.getIndex(word)].size()).normal_(-0.05,0.05)
            torch.save(emb, emb_file)
            print('done creating emb, quit')

        emb = emb.cuda()

        self.embedding_model.state_dict()['weight'].copy_(emb)

        return model.cuda()

