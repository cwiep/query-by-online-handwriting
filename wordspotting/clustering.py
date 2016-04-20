"""
@author: Christian Wieprecht
@license: Apache License, Version 2.0
"""

import numpy as np
from scipy.cluster.vq import kmeans2


def cluster(descriptors, n_centroids):
    """
    Cluster all feature vectors in descriptors into codebook of size n_centroids.

    @param descriptors: n_samples x n_features matrix
    @param n_centroids: number of clusters to be created
    @return: codebook, labels
    """
    print("Extracting codebook of size {} from {} descriptors".format(n_centroids, len(descriptors)))

    # needs vlfeat, faster for integer vectors, note that the input matrix n_features x n_samples
    # codebook, labels = vlfeat.vl_ikmeans(np.array(descriptors).T, n_centroids, max_niters=20, verbose=1)

    codebook, labels = kmeans2(np.array(descriptors), n_centroids, iter=20, minit='points')
    return codebook, labels