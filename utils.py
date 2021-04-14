from scipy.spatial import cKDTree

def nearest_neighbours(data, query_data, k):
	kdtree = cKDTree(locs)
	return (kdtree.query(query_locs, k)[1]).T