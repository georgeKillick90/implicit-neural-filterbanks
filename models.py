import torch
import torch.nn as nn
from conv import ConvNN
from pooling import AvgPoolNN, MaxPoolNN




class ResBlockNN(nn.Module):
	def __init__(self, in_channels, out_channels, locs_in, locs_out):
		super(ResBlockNN, self).__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.locs_in = locs_in
		self.locs_out = locs_out

		self.conv1 = ConvNN(in_channels, out_channels, 9, self.locs_in, self.locs_out)
		self.conv2 = ConvNN(out_channels, out_channels, 9, self.locs_out, self.locs_out)

		self.bn1 = nn.BatchNorm1d(out_channels)
		self.bn2 = nn.BatchNorm1d(out_channels)
		self.bn3 = None

		self.activation = nn.ReLU(inplace=True)

		self.expand = None

		if in_channels != out_channels:
			self.expand = nn.Conv1d(in_channels, out_channels, 1)
			self.bn3 = nn.BatchNorm1d(out_channels)


	def forward(self, x):
		identity = x
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.activation(out)

		out = self.conv2(out)
		out = self.bn2(out)

		if self.expand != None:
			identity = self.bn3(self.expand(x))

		out += identity
		out = self.activation(out)

		return out
