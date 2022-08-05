import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Config(object):

    def __init__(self, dataset, embedding):
        self.allfile_path = dataset + '/data/ATC_SMILES.csv'
        self.model_name = 'TextCNN'
        self.vocab_path = dataset + '/data/SmiVocab.pkl'
        self.save_path = dataset + '/data/' + self.model_name + '.ckpt'
        self.log_path = dataset + '/log/' + self.model_name
        self.embedding_pretrained = torch.tensor(
            np.load(dataset + '/data/embedding_SMILES_Vocab.npz')["embeddings"].astype('float32'))\
            if embedding != 'random' else None
        self.device = torch.device('cuda:0')
        self.dropout = 0.2
        self.require_improvement = 60000
        self.num_classes = 14
        self.n_vocab = 110
        self.num_epochs = 20
        self.batch_size = 16
        self.pad_size = 787
        self.learning_rate = 1e-3
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300
        self.filter_sizes = (2, 4, 6, 8, 10, 16, 24)
        self.num_filters = 256
        self.K=4545

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            #print('Use embedding_pretrained')
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embed)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = self.embedding(x[0])
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out

