from scipy.spatial import cKDTree
import pandas as pd
import torch
import numpy as np


def nearest_neighbours(data, query_data, k):
	kdtree = cKDTree(data)
	return (kdtree.query(query_data, k)[1]).T

def load_meta_data(textfile):
  	meta_data = pd.read_csv(textfile, header=0)
  	return meta_data.values.tolist()

class RetinaTransform(object):
    def __init__(self, retina, size, fixation=None, backproject=True):

        self.backproject = backproject
        self.retina = retina
        self.size = size

        if (fixation == None):
        	self.fixation = (size[0] / 2. , size[1] / 2.)
        else:
        	self.fixation = fixation

        self.retina.prepare(self.size, self.fixation)

    def __call__(self, sample):
        sample = self.retina.sample(sample.permute(1,2,0).numpy() * 255., self.fixation)

        if (self.backproject):
          sample = self.retina.backproject_last()
        
        return torch.tensor(sample.T).float() / 255.

def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''

    # From SIREN exploration notebook

    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid

def cart2pol(coords):

    # Cartesian coordinates to polar.

    x = coords[:,0]
    y = coords[:,1]
    rho = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    return np.stack([theta, rho],1)

def pol2cart(coords):

    # Polar coordinates to cartesian.

    theta = coords[:,0]
    rho = coords[:,1]
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)

    return np.stack([x, y], 1)