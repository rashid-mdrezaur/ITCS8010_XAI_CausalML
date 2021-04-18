#!/usr/bin/env python3

from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_feats, hidden_dim, n_classes, dropout=0.2):
        super(MLP, self).__init__()

        self.hidden_layer = nn.Linear(in_feats, hidden_dim)
        self.out_layer = nn.Linear(hidden_dim, n_classes)
        self.dropout = nn.Dropout(dropout)
        self.batchnorm = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        x = self.dropout(x)
        x = self.hidden_layer(x)
        x = F.relu(self.batchnorm(x))
        x = self.out_layer(x)

        return x
