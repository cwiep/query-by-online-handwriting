# -*- coding: utf-8 -*-
"""
@author: Christian Wieprecht
@license: Apache License, Version 2.0

Generator for novel bag-of-online-features representation of online-handwritten queries.
"""
import numpy as np
import tools.logging as log


class BoofGenerator():
    def __init__(self, spatial_pyramid):
        self.spatial_pyramid = spatial_pyramid

    def build_feature_vectors_matrix(self, keypoints, labels):
        log.d("Building trajectory feature matrix...")
        num_features = self.spatial_pyramid.descriptor_size()
        num_examples = len(keypoints)
        feat_mat = np.zeros(shape=(num_examples, num_features))
        i = 0
        for keyp, lab in zip(keypoints, labels):
            feat_mat[i] = self.__build_feature_vector(keyp, lab)
            i += 1
            log.update_progress(i+1, num_examples)
        print("")
        log.d("Accumulated {} feature vectors.".format(len(feat_mat)))
        return np.array(feat_mat)

    def __build_feature_vector(self, keypoints, labels):
        minx = min(keypoints[:, 0])
        maxx = max(keypoints[:, 0])
        miny = min(keypoints[:, 1])
        maxy = max(keypoints[:, 1])
        return self.spatial_pyramid.calculate_descriptor(keypoints, labels, (minx, miny), maxx-minx, maxy-miny)
