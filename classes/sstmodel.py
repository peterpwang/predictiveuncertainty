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
from  torchtext.datasets.sst import SST
from ignite.utils import convert_tensor
import treelstm


SEED = 1234
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

def _label_node_index(node, n=0):
    node['index'] = n
    for child in node['children']:
        n += 1
        _label_node_index(child, n)

def _gather_node_attributes(node, key):
    features = [node[key]]
    for child in node['children']:
        features.extend(_gather_node_attributes(child, key))
    return features

def _gather_adjacency_list(node):
    adjacency_list = []
    for child in node['children']:
        adjacency_list.append([node['index'], child['index']])
        adjacency_list.extend(_gather_adjacency_list(child))

    return adjacency_list

def convert_tree_to_tensors(tree, device=torch.device('cpu')):
    # Label each node with its walk order to match nodes to feature tensor indexes
    # This modifies the original tree as a side effect
    _label_node_index(tree)

    features = _gather_node_attributes(tree, 'features')
    labels = _gather_node_attributes(tree, 'labels')
    adjacency_list = _gather_adjacency_list(tree)

    node_order, edge_order = calculate_evaluation_orders(adjacency_list, len(features))

    return {
        'features': torch.tensor(features, device=device, dtype=torch.float32),
        'labels': torch.tensor(labels, device=device, dtype=torch.float32),
        'node_order': torch.tensor(node_order, device=device, dtype=torch.int64),
        'adjacency_list': torch.tensor(adjacency_list, device=device, dtype=torch.int64),
        'edge_order': torch.tensor(edge_order, device=device, dtype=torch.int64),
    }


# Abstract SST model
class AbstractSSTTextClassificationModel(AbstractClassificationModel):

    def __init__(self, args):
        super(AbstractSSTTextClassificationModel, self).__init__(args)
        #spacy.prefer_gpu()
        #nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser","ner"])
        #self.TEXT = data.Field(tokenize=nlp, batch_first=True, include_lengths=True)
        self.TEXT = data.Field(lower=True, batch_first=True)
        self.LABEL = data.LabelField(dtype = torch.long, batch_first=True)
        self.prepare_batch = text_prepare_batch
    
    # compile model
    def compile_model(self, net):
        print("Learning rate is set to ", self.learning_rate)
        #define optimizer and loss
        optimizer = optim.Adam(net.parameters())
        criterion = nn.BCELoss().cuda(0)
        return optimizer, criterion

    # Load dataset and split into training and test sets.
    def load_dataset(self):

        fields = [('label', self.LABEL),('text',self.TEXT),('text',self.TEXT)]

        #loading custom dataset
        train_data = torchtext.datasets.SST('data/sst/trees/train.txt', self.TEXT, self.LABEL, subtrees=True)
        validation_data = torchtext.datasets.SST('data/sst/trees/dev.txt', self.TEXT, self.LABEL, subtrees=True)
        test_data = torchtext.datasets.SST('data/sst/trees/test.txt', self.TEXT, self.LABEL, subtrees=True)

        #print preprocessed text
        print(vars(train_data.examples[0]))

        #train_data, validation_data = train_data.split(split_ratio=0.8)

        #train_data = convert_tree_to_tensors(train_data)
        #validation_data = convert_tree_to_tensors(validation_data)
        #test_data = convert_tree_to_tensors(test_data)

        self.TEXT.build_vocab(train_data,min_freq=3,vectors = "glove.6B.100d")  
        self.LABEL.build_vocab(train_data)

        #No. of unique tokens in text
        #print("Size of TEXT vocabulary:",len(self.TEXT.vocab))

        #No. of unique tokens in label
        #print("Size of LABEL vocabulary:",len(self.LABEL.vocab))

        #Commonly used words
        #print(self.TEXT.vocab.freqs.most_common(10))  

        #Word dictionary
        #print(self.TEXT.vocab.stoi)   

        #set batch size
        BATCH_SIZE = 64

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        #Load an iterator
        train_iterator, validation_iterator = data.BucketIterator.splits(
            (train_data, validation_data), 
            batch_size = BATCH_SIZE,
            sort_key = lambda x: len(x.text),
            sort_within_batch=True,
            device = device)
        test_iterator = data.BucketIterator(test_data, 
            batch_size = BATCH_SIZE,
            sort_key = lambda x: len(x.text),
            sort_within_batch=True,
            device = device)

        return train_iterator, validation_iterator, test_iterator


# Prepare_batch function for text classification
def text_prepare_batch(batch, device=None, non_blocking=False):
    """Prepare batch for training: pass to a device with options.

    """
    x, y = batch.text, batch.label
    return (convert_tensor(x, device=device, non_blocking=non_blocking),
            convert_tensor(y, device=device, non_blocking=non_blocking))


# Tree LSTM model
class TreeLSTMNet(AbstractSSTTextClassificationModel):
    
    # Set model
    def define_model(self):

        model = treelstm.TreeLSTM(2, 1).train()
        return model.cuda()

