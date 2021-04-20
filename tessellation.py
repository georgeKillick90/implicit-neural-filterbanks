import numpy as np
from utils import pol2cart, cart2pol

# Author: George Killick

def fibonacci_sunflower(n_nodes):
	""" Generates points using the golden ratio
		Parameters
		----------
		n_nodes: number of points to be generated
		Return: numpy array of points
	"""

	# Golden ratio (can use any value here for different
	# tessellation mosaics)
	g_ratio = (np.sqrt(5) + 1) / 2

	nodes = np.arange(1,n_nodes+1)

	# calculate rho for each point
	# sqrt to maintain uniform density
	# offset by -0.5 to fill centre gap
	# normalize by sqrt of total nodes to constrain to unit circle
	rho = np.sqrt(nodes-0.5)/np.sqrt(n_nodes)

	# Rotate each point by the golden ratio
	# being the most unrationable number means the points get
	# optimal spacing
	theta = np.pi * 2 * g_ratio * nodes

	# convert to cartesian coordinates
	x = rho * np.cos(theta)
	y = rho * np.sin(theta)

	# return points in standard format
	return np.array([x,y]).T

def fibonacci_retina(n_nodes, fovea, foveal_density):

	""" Generates points using the fibonacci sunflower
		and dilates them with the dilate function found in utils.
		See README for more description of this dilate function.
		Parameters
		----------
		n_nodes: number of nodes in tessellation
		fovea: size of foveal region in tessellation; 0 < fovea <= 1
		fovea_density: scaling factor to affect the ratio of nodes
		in and outside the fovea.
		Return: numpy array of points
	"""
	x = fibonacci_sunflower(n_nodes)

	x = normalize(x)
	x = cart2pol(x)
	x[:,1] *= (1/(fovea + ((2*np.pi*fovea)/foveal_density)) ** x[:,1] ** foveal_density)
	x = pol2cart(x)

	return normalize(x)


def normalize(points):

    # Constrains the retina tessellation to a unit
    # circle.

    points = cart2pol(points)
    points[:,1] /= np.max(points[:,1])
    points = pol2cart(points)

    return points