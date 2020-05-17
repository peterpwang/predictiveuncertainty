from .abstractmodel import AbstractImageClassificationModel

import os
import random

import torch
from torchtext import data    
import torch.nn as nn
import torch.nn.functional as F


# Abstract 20 newsgroups model
class AbstractTwentyNewsGroupsTextClassificationModel(AbstractImageClassificationModel):

    # Load dataset and split into training and test sets.
    def load_dataset(self):

        TEXT = data.Field(tokenize='spacy', batch_first=True, include_lengths=True)
        LABEL = data.LabelField(dtype = torch.float, batch_first=True)

        fields = [('label', LABEL),('text',TEXT),('text',TEXT)]

        #loading custom dataset
        training_data = data.TabularDataset(path = 'data/20newsgroups/train.csv', format='csv', fields=fields, skip_header = False)
        test_data = data.TabularDataset(path = 'data/20newsgroups/test.csv', format='csv', fields=fields, skip_header = False)

        #print preprocessed text
        print(vars(training_data.examples[0]))

        train_data, validation_data = training_data.split(split_ratio=0.7, random_state = random.seed(SEED))

        #initialize glove embeddings
        TEXT.build_vocab(train_data,min_freq=3,vectors = "glove.6B.100d")  
        LABEL.build_vocab(train_data)

        #No. of unique tokens in text
        print("Size of TEXT vocabulary:",len(TEXT.vocab))

        #No. of unique tokens in label
        print("Size of LABEL vocabulary:",len(LABEL.vocab))

        #Commonly used words
        print(TEXT.vocab.freqs.most_common(10))  

        #Word dictionary
        print(TEXT.vocab.stoi)   

        #set batch size
        BATCH_SIZE = 64

        #Load an iterator
        train_iterator, validation_iterator = data.BucketIterator.splits(
            (train_data, validation_data), 
            batch_size = BATCH_SIZE,
            sort_key = lambda x: len(x.text),
            sort_within_batch=True,
            device = device)
        test_iterator = data.BucketIterator.splits(test_data, 
            batch_size = BATCH_SIZE,
            sort_key = lambda x: len(x.text),
            sort_within_batch=True,
            device = device)

        return train_iterator, validation_iterator, test_iterator


