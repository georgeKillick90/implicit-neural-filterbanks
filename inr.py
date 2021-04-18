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
        self.fc1 = SineLayer(in_features, hidden_size, is_first=True)
        self.fc2 = SineLayer(hidden_size, hidden_size,)
        self.fc3 = SineLayer(hidden_size, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
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

class SineLayer(nn.Module):
    '''
    Taken from https://vsitzmann.github.io/siren/
    Only here currently for ease of use while researching, will remove at a later data
    '''

    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))