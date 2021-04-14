#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author georgekillick90
"""

import torch
import torch.nn as nn

class INR(nn.Module):
    def __init__(self, in_features, out_features, hidden_size):
        super(INR, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, out_features)

    def forward(self, x):
        x = torch.sin(self.fc1(x))
        x = torch.sin(self.fc2(x))
        x = self.fc3(x)
        return x

class INFilterBank(nn.Module):
    def __init__(self, in_channels, out_channels, in_features=2, hidden_size=9):
        super(INFilterBank, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.out_features = in_channels * out_channels
        self.inr = INR(in_features, self.out_features, hidden_size)

    def forward(self, x):
        x = self.inr(x)
        # reshape to an appropriate shape so that it can be used as a filter
        x = torch.reshape(x, shape=(-1, self.out_channels, self.in_channels))
        return x