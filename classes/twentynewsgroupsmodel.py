from .abstractmodel import AbstractImageClassificationModel

import os
import random
import spacy

import torch
from torchtext import data    
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ignite.utils import convert_tensor


SEED = 1234
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# Abstract 20 newsgroups model
class AbstractTwentyNewsGroupsTextClassificationModel(AbstractImageClassificationModel):

    def __init__(self, args):
        super(AbstractTwentyNewsGroupsTextClassificationModel, self).__init__(args)
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
        criterion = nn.CrossEntropyLoss().cuda(0)
        return optimizer, criterion

    # Load dataset and split into training and test sets.
    def load_dataset(self):

        fields = [('label', self.LABEL),('text',self.TEXT),('text',self.TEXT)]

        #loading custom dataset
        training_data = data.TabularDataset(path = 'data/20newsgroups/train.csv', format='csv', fields=fields, skip_header = False)
        test_data = data.TabularDataset(path = 'data/20newsgroups/test.csv', format='csv', fields=fields, skip_header = False)

        #print preprocessed text
        print(vars(training_data.examples[0]))

        train_data, validation_data = training_data.split(split_ratio=0.8)

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


# TextCNN model
class TestNet(nn.Module):
    
    def __init__(self, text, vocab_size, embedding_dim, kernel_sizes, num_filters, num_classes, d_prob, mode):
        super(TestNet, self).__init__()
        self.text = text
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.kernel_sizes = kernel_sizes
        self.num_filters = num_filters
        self.num_classes = num_classes
        self.d_prob = d_prob
        self.mode = mode
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=1)
        self.load_embeddings()
        self.conv = nn.ModuleList([nn.Conv1d(in_channels=embedding_dim,
                                             out_channels=num_filters,
                                             kernel_size=k, stride=1) for k in kernel_sizes])
        self.dropout = nn.Dropout(d_prob)
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, num_classes)

    def forward(self, x):
        batch_size, sequence_length = x.shape
        x = self.embedding(x).transpose(1, 2)
        x = [F.relu(conv(x)) for conv in self.conv]
        x = [F.max_pool1d(c, c.size(-1)).squeeze(dim=-1) for c in x]
        x = torch.cat(x, dim=1)
        x = self.fc(self.dropout(x))
        return torch.sigmoid(x).squeeze()

    def load_embeddings(self):
        if 'static' in self.mode:
            self.embedding.weight.data.copy_(self.text.vocab.vectors)
            if 'non' not in self.mode:
                self.embedding.weight.data.requires_grad = False
                print('Loaded pretrained embeddings, weights are not trainable.')
            else:
                self.embedding.weight.data.requires_grad = True
                print('Loaded pretrained embeddings, weights are trainable.')
        elif self.mode == 'rand':
            print('Randomly initialized embeddings are used.')
        else:
            raise ValueError('Unexpected value of mode. Please choose from static, nonstatic, rand.')


# Test 20 Newsgroups model
class TestTwentyNewsgroupsModel(AbstractTwentyNewsGroupsTextClassificationModel):

    # Set model
    def define_model(self):

        #define hyperparameters
        vocab_size, embedding_dim = self.TEXT.vocab.vectors.shape

        model = TestNet(self.TEXT,
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                kernel_sizes=[3, 4, 5],
                num_filters=100,
                num_classes=4, 
                d_prob=0.5,
                mode='static')

        return model.cuda()
