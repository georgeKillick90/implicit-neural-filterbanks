from scipy.spatial import cKDTree

def nearest_neighbours(data, query_data, k):
	kdtree = cKDTree(data)
	return (kdtree.query(query_data, k)[1]).T