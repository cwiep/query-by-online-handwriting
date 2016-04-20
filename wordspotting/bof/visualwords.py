# -*- coding: utf-8 -*-
"""
@author: Christian Wieprecht
@license: Apache License, Version 2.0
"""

import numpy as np
import vlfeat

from wordspotting import clustering


def calculate_sift_descriptors(image, step_size, cell_size, norm_threshold=0.0):
    """
    Calculate sift descriptors on single image.

    @param image: Image to extract sift descriptors from.
    @param step_size: Horizontal and vertical distance between keypoints in pixels.
    @param cell_size: Area size of one descriptor in pixels.
    @return: Coordinates and descriptors for each keypoint.
    """
    # we don't want to start directly in the corner of the page
    if cell_size == 8:
        off = 2.5
    elif cell_size == 5:
        off = 7.5
    else:
        off = 0.0

    frames, desc = vlfeat.vl_dsift(image,
                                   bounds=np.array((off, off, image.shape[0]-off, image.shape[1]-off), 'f'),
                                   step=step_size,
                                   size=cell_size,
                                   norm=(norm_threshold > 0.0))

    frames = frames.T
    desc = desc.T

    # throw away all descriptors with a magnitude < norm_threshold
    if norm_threshold > 0.0:
        norms = frames[:, 2]
        frames = np.array([p[:2] for i, p in enumerate(frames) if norms[i] > norm_threshold])
        desc = np.array([d for i, d in enumerate(desc) if norms[i] > norm_threshold])
    return frames, desc


def calculate_visual_words_from_images(images, n_centroids, step_size, cell_size, norm_threshold=0.0):
    """
    Extract SIFT descriptors from each page. Calculate n_centroid visual words by clustering descriptors.

    @param images: Images (e.g. matplotlib.image) to extract SIFT descriptors from.
    @param n_centroids: Number of visual words.
    @param step_size: Distance (both horizontally and vertically) between two descriptor centers.
    @param cell_size: List of cell sizes of a descriptor. Complete descriptor has size 4 * cell_size.
    @return: visual words, keypoints-coordinates, assigned visual word for each keypoint, number of keypoints per page
    """
    all_desc = []
    keypoints = []
    page_sizes = []
    # calculate keypoints and sift descriptors for each page
    for image in images:
        im_arr = np.asarray(image, dtype='float32')
        k, d = calculate_sift_descriptors(im_arr, step_size, cell_size, norm_threshold)
        print("Extracted {} descriptors.".format(len(k)))
        all_desc.extend(d)
        keypoints.extend(k)
        page_sizes.append(len(k))

    visualwords, labels = clustering.cluster(all_desc, n_centroids)
    return visualwords, keypoints, labels, np.array(page_sizes)