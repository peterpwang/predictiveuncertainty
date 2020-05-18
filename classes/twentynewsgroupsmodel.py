from .abstractmodel import AbstractImageClassificationModel

import os
import random
import spacy

import torch
from torchtext import data    
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# Abstract 20 newsgroups model
class AbstractTwentyNewsGroupsTextClassificationModel(AbstractImageClassificationModel):

    def __init__(self, args):
        spacy.prefer_gpu()
        nlp = spacy.load("en_core_web_sm")
        self.TEXT = data.Field(tokenize=nlp, batch_first=True, include_lengths=True)
        self.LABEL = data.LabelField(dtype = torch.float, batch_first=True)
        super(AbstractTwentyNewsGroupsTextClassificationModel, self).__init__(args)
    
    # compile model
    def compile_model(self, net):
        print("Learning rate is set to ", self.learning_rate)
        #define optimizer and loss
        optimizer = optim.Adam(model.parameters())
        criterion = nn.BCELoss()
        return optimizer, criterion

    # Load dataset and split into training and test sets.
    def load_dataset(self):


        fields = [('label', self.LABEL),('text',self.TEXT),('text',self.TEXT)]

        #loading custom dataset
        training_data = data.TabularDataset(path = 'data/20newsgroups/train.csv', format='csv', fields=fields, skip_header = False)
        test_data = data.TabularDataset(path = 'data/20newsgroups/test.csv', format='csv', fields=fields, skip_header = False)

        #print preprocessed text
        print(vars(training_data.examples[0]))

        train_data, validation_data = training_data.split(split_ratio=0.7, random_state = random.seed(SEED))

        #initialize glove embeddings
        self.TEXT.build_vocab(train_data,min_freq=3,vectors = "glove.6B.100d")  
        self.LABEL.build_vocab(train_data)

        #No. of unique tokens in text
        print("Size of TEXT vocabulary:",len(self.TEXT.vocab))

        #No. of unique tokens in label
        print("Size of LABEL vocabulary:",len(self.LABEL.vocab))

        #Commonly used words
        print(self.TEXT.vocab.freqs.most_common(10))  

        #Word dictionary
        print(self.TEXT.vocab.stoi)   

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


# Test model, quick and simple
class TestNet(nn.Module):
    
    #define all the layers used in model
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        
        #Constructor
        super().__init__()     
        
        #embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        #lstm layer
        self.lstm = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=bidirectional, 
                           dropout=dropout,
                           batch_first=True)
        
        #dense layer
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
        #activation function
        self.act = nn.Sigmoid()
        
    def forward(self, text, text_lengths):
        
        #text = [batch size,sent_length]
        embedded = self.embedding(text)
        #embedded = [batch size, sent_len, emb dim]
      
        #packed sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths,batch_first=True)
        
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        #hidden = [batch size, num layers * num directions,hid dim]
        #cell = [batch size, num layers * num directions,hid dim]
        
        #concat the final forward and backward hidden state
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
                
        #hidden = [batch size, hid dim * num directions]
        dense_outputs=self.fc(hidden)

        #Final activation function
        outputs=self.act(dense_outputs)
        
        return outputs


# Test 20 Newsgroups model
class TestTwentyNewsgroupsModel(AbstractTwentyNewsGroupsTextClassificationModel):

    # Set model
    def define_model(self):
        #define hyperparameters
        size_of_vocab = len(self.TEXT.vocab)
        embedding_dim = 100
        num_hidden_nodes = 32
        num_output_nodes = 1
        num_layers = 2
        bidirection = True
        dropout = 0.2

        #instantiate the model
        model = TestNet(size_of_vocab, embedding_dim, num_hidden_nodes,num_output_nodes, num_layers, 
                   bidirectional = True, dropout = dropout)

        #Initialize the pretrained embedding
        pretrained_embeddings = self.TEXT.vocab.vectors
        model.embedding.weight.data.copy_(pretrained_embeddings)
        return model.cuda()
