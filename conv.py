import torch
import torch.nn as nn
from inr import INFilterBank
from utils import nearest_neighbours

class ConvNN(nn.Module):
    def __init__(self, in_channels, out_channels, k, locs_in, locs_out):
        super(ConvNN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k
        self.locs_in = locs_in 
        self.locs_out = locs_out
        self.n_samples = len(locs_out)

        # Gets the nearest neighbours relationships
        self.neighbours = nearest_neighbours(locs_in, locs_out, k)

        # Unfold based on nearest neighbours
        self.locs_unfold = self.locs_in.T[:, self.neighbours]

        # Normalize each unfolded slice by the centre of its receptive field
        centres = self.locs_out.T.unsqueeze(1)
        self.locs_unfold = self.locs_unfold - centres

        # find the radius of the receptive field for each unfold slice and normalize by radius
        # to account for space variant sampling
        radius, _ = torch.max(torch.linalg.norm(self.locs_unfold, dim=0), dim=0)
        self.locs_unfold = self.locs_unfold / radius.unsqueeze(0)
        self.locs_unfold = self.locs_unfold.reshape(shape=(2, -1))

        # Construct an implicit neural representation filterbank
        self.inr_filters = INFilterBank(self.in_channels, self.out_channels)

        # register buffers
        self.register_buffer('_neighbours', torch.tensor(self.neighbours))
        self.register_buffer('_locs_unfold', torch.tensor(self.locs_unfold).float())

        # register parameter so sent to device
        self.register_parameter('bias', nn.Parameter(torch.zeros(out_channels).unsqueeze(0).unsqueeze(-1))) 

    def forward(self, x):

        # Unfold similar to pytorch unfold but instead we are using nearest neighbours
        x_unfold = x[:,:,self._neighbours]

        # Reshape the filter weights back to the appropriate shape.
        filter_values = self.inr_filters(self._locs_unfold.T).permute(1, 2, 0) 
        filter_values = filter_values.reshape(self.out_channels, self.in_channels, self.k, self.n_samples)

        # Einsum to perform the convolution, matmul maybe faster, for now fine.
        x = torch.einsum('...jkl,ijkl->...il', x_unfold, filter_values)

        # add bias
        x = x + self.bias

        return x

