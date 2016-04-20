# -*- coding: utf-8 -*-
"""
@author: Christian Wieprecht
@license: Apache License, Version 2.0

Building bag-of-features matrices.
"""
import math
import numpy as np
import tools.logging as log


class BofGenerator():
    def __init__(self, spatial_pyramid):
        self.spatial_pyramid = spatial_pyramid

    def build_feature_vectors_matrix(self, trans_data_pages, keypoint_data_pages, label_data_pages, step_size):
        """
        Builds feature matrix containing one row per word image and one column per feature.

        @param trans_data_pages: List of pages. Each page is a list containing TransData objects for each word.
        @param keypoints: List of all keypoints (concatenated from each page).
        @param labels: List of all labels (concatenated from each page).
        @param step_size: Distance between keypoints.
        @return: Matrix containing one row per word and one column per feature.
        """
        label_matrices = self.__build_pages_matrices(keypoint_data_pages, label_data_pages)
        num_pages = len(label_matrices)
        num_rows = sum([len(t) for t in trans_data_pages])
        feat_mat = np.zeros(shape=(num_rows, self.spatial_pyramid.descriptor_size()))
        i = 0
        for page_idx in range(num_pages):
            keypoints = keypoint_data_pages[page_idx]
            label_matrix = label_matrices[page_idx]
            for word_data in trans_data_pages[page_idx]:
                origin = keypoints[0]
                x = math.ceil((word_data.xstart - origin[0]) / float(step_size))
                y = math.ceil((word_data.ystart - origin[1]) / float(step_size))
                dx = math.floor(word_data.width / float(step_size))
                dy = math.floor(word_data.height / float(step_size))
                # in terms of matrix notation, y is row here!
                # +1 is necessary, because in python 1:5 is 1,2,3,4!
                desc_mat = label_matrix[int(y):int(y+dy+1), int(x):int(x+dx+1)]
                # print(desc_mat.shape)
                visual_descriptor = self.spatial_pyramid.calculate_descriptor_from_mat(desc_mat)
                feat_mat[i] = visual_descriptor
                # fw_matrix[i] = mathutils.normalize(fw_matrix[i])
                i += 1
        log.d("Visual feature-matrix has shape {}".format(feat_mat.shape))
        return feat_mat

    def __build_pages_matrices(self, keypoint_data_pages, label_data_pages):
        """
        Build matrix for each page, representing a grid of labels of keypoints.

        @param keypoints: List of all keypoints.
        @param labels: List of all visual word labels assigned to keypoints.
        @return: List of matrices, containing labels/keypoints for each page.
        """
        pages = []
        for keypoints, labels in zip(keypoint_data_pages, label_data_pages):
            pages.append(self.__build_page_matrix(keypoints, labels))
        return pages

    def __build_page_matrix(self, keypoints, labels):
        page_size = len(keypoints)
        num_keyp_per_col = 0
        i = 0
        while keypoints[i][0] == keypoints[0][0]:
            num_keyp_per_col += 1
            i += 1
        num_keyp_per_row = page_size / num_keyp_per_col
        # log.d("Matrix shape for page {}: {}".format(pid, (num_keyp_per_col, num_keyp_per_row)))
        # log.d("Image shape {}".format(images[pid].shape))
        return np.reshape(labels, (num_keyp_per_col, num_keyp_per_row), order='F')