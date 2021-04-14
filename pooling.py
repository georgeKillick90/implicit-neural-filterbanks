import torch
import torch.nn as nn
from inr import INFilterBank
from utils import nearest_neighbours


class PoolNU(nn.Module):
    def __init__(self, locs_in, locs_out, k, method):
        super(PoolNU, self).__init__()

        # I am thinking that the way locs are passed are important, copy might be necessary
        self.locs_in = locs_in
        self.locs_out = locs_out
        self.method = method
        self.k = k
        self.neighbours = nearest_neighbours(locs_in, locs_out, k)

        # register buffers
        self.register_buffer('_neighbours', torch.tensor(self.neighbours))


    def forward(self, x):
        # Unfold similar to pytorch unfold but instead we are using nearest neighbours
        x = x[:,:,self._neighbours]
        # Max Pool, at some point I'll extend this to do average pooling
        # also returning indices as well might be useful
        x = torch.max(x, dim=-2)[0]
        return x