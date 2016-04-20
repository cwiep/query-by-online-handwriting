# -*- coding: utf-8 -*-
"""
@author: Christian Wieprecht
@license: Apache License, Version 2.0
"""

from scipy.cluster.vq import vq


def quantize_descriptors(descriptors, codebook):
    """
    Assigns each descriptor to its nearest visual word.

    @param descriptors: Descriptors (feature vectors).
    @param codebook: Precalculated codebook (n_clusters x n_features)
    @return: Array of labels in the same order as desc
    """
    labels, _ = vq(descriptors, codebook)
    return labels