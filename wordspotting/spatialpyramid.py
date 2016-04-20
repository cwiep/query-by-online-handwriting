# -*- coding: utf-8 -*-
"""
@author: Christian Wieprecht
@license: Apache License, Version 2.0
"""
import numpy as np

from tools import mathutils


class SpatialPyramid():
    def __init__(self, levels, num_centroids):
        self.levels = levels
        self.num_centroids = num_centroids

    def num_total_bins(self):
        return sum([level[0]*level[1] for level in self.levels])

    def descriptor_size(self):
        return self.num_total_bins() * self.num_centroids

    def calculate_descriptor_from_mat(self, desc_mat):
        """
        Calculates descriptor for given matrix of codebook indices (labels) by applying
        a spatial pyramid scheme and flattening and normalizing the resulting array.

        @return: One-dimensional array representing descriptor of the given input matrix.
        """
        spatial_pyramid = self.__extract_spatial_pyramid_from_mat(desc_mat)
        return self.__pyramid_to_descriptor(spatial_pyramid)

    def __extract_spatial_pyramid_from_mat(self, desc_mat):
        """
        Applies a spatial pyramid scheme to a given matrix of codebook indices (labels).

        @return: An array of arrays containing each bin for each level of the spatial pyramid.
        """
        spatial_pyramid = []
        for level in self.levels:
            num_bins_x, num_bins_y = level
            bins = np.zeros(shape=(num_bins_y, num_bins_x), dtype=object)
            bin_size_x = desc_mat.shape[1]/num_bins_x
            bin_size_y = desc_mat.shape[0]/num_bins_y
            # log.d("Binsize on level {}: {}, {}".format(level, bin_size_x, bin_size_y))
            for x in range(num_bins_x):
                for y in range(num_bins_y):
                    # log.d("level {} {}:".format(x, y))
                    # If it's the last bin in x- or y-direction we use all the remaining
                    # points and ignore bin size.
                    xmin = x * bin_size_x
                    xmax = desc_mat.shape[1]-1 if x == num_bins_x - 1 else (x+1)*bin_size_x
                    ymin = y * bin_size_y
                    ymax = desc_mat.shape[0]-1 if y == num_bins_y - 1 else (y+1)*bin_size_y
                    indices = np.ravel(desc_mat[ymin:ymax, xmin:xmax])
                    hist = np.bincount(indices, minlength=self.num_centroids)
                    bins[y, x] = hist
            spatial_pyramid.append(bins)
        return np.array(spatial_pyramid)

    def calculate_descriptor(self, keypoints, labels, origin, width, height):
        """
        Calculates visual descriptor for a given list of keypoints/labels by applying
        a spatial pyramid scheme and flattening and normalizing the resulting array.

        NOTE: This is slower than calculate_visual_descriptor_from_mat but will work on
        data that is not in matrix form.

        @param keypoints: Keypoints (with format [x,y]) for which labels are given.
        @param labels: labels[i] contains the visual word label of the i-th keypoint.
        @param origin: Origin of the area that the spatial pyramid is calculated for (e.g. (0,0) for whole image).
        @param width: Width of the area the spatial pyramid is calculated for.
        @param height: Height of the area the spatial pyramid is calculated for.
        @return: An array of arrays containing each bin for each level of the spatial pyramid.
        """
        spatial_pyramid = self.__extract_spatial_pyramid(keypoints, labels, origin, width, height)
        return self.__pyramid_to_descriptor(spatial_pyramid)

    def __extract_spatial_pyramid(self, keypoints, labels, origin, width, height):
        """
        Applies a spatial pyramid scheme to a given set of keypoints/labels.
        """
        levels = len(self.levels)
        spatial_pyramid = []
        hist_template = [0] * self.num_centroids
        for level in range(levels):
            num_bins_x = self.levels[level][0]
            num_bins_y = self.levels[level][1]
            bins = [[list(hist_template) for _ in range(num_bins_y)] for _ in range(num_bins_x)]
            bin_size_x, bin_size_y = float(width)/num_bins_x, float(height)/num_bins_y
            # log.d("Binsize on level {}: {}, {}".format(level, bin_size_x, bin_size_y))
            for index, point in enumerate(keypoints):
                # bin_size + 1: preventing index error, for keypoints situated directly on an edge
                # of a bin: keypoint (82, 176), bin_sizes (405, 176) would yield ybin-index 1 (of total 1)
                xbin = int((point[0]-origin[0]) / (bin_size_x+1))
                ybin = int((point[1]-origin[1]) / (bin_size_y+1))
                bins[xbin][ybin][labels[index]] += 1
            spatial_pyramid.append(bins)
        return np.array(spatial_pyramid)

    def __pyramid_to_descriptor(self, descriptor_pyramid):
        """
        Calculates descriptor by flattening the spatial bins in descriptor_pyramid.

        @return: One-dimensional array representing visual descriptor of a word-snippet.
        """
        flattened = []
        for index, level in enumerate(descriptor_pyramid):
            tmp = []
            for horiz_bin in level:
                for vert_bin in horiz_bin:
                    tmp.extend(mathutils.normalize(vert_bin))
            flattened.extend(tmp)
        return mathutils.normalize(np.array(mathutils.power_normalize(flattened)))
