#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Christian Wieprecht
@license: Apache License, Version 2.0

Read {FILENAMES}.png from PAGE_FOLDER. Extract sift descriptors and
build a codebook by clustering. Quantize the descriptors on
each page with that codebook. Save codebook and one file for each
page containing the keypoints and the codebook indices (labels).
"""

import matplotlib.image as mimg
import numpy as np

from tools import mathutils
import tools.logging as log
import wordspotting.bof.visualwords as vw


STEP_SIZE = 5
CELL_SIZE = 10
N_CENTROIDS = 2048
FILENAMES = ['2700270']#, '2710271', '2720272', '2730273', '2740274',
            #'2750275', '2760276', '2770277', '2780278', '2790279',
            #'3000300', '3010301', '3020302', '3030303', '3040304',
            #'3050305', '3060306', '3070307', '3080308', '3090309']
PAGE_FOLDER = "/home/chris/Work/MA.data/gw_ls12/pages"
# PAGE_FOLDER = "/vol/bof/lrothack/gw-wordspotting/Data/pages"


def extract_vw():
    print("Loading images and extracting visual words for")
    for f in FILENAMES:
        print("{}/{}.png".format(PAGE_FOLDER, f))

    """
    If you want to use an existing codebook to calculate descriptors, first
    load the codebook

    visualwords = np.load(VISUAL_WORDS_PATH)

    and then use something like the following for every image file f:

    image = mimg.imread("{}/{}".format(PAGE_FOLDER, f))

    print("Extracting descriptors from '{}'".format(f))
    k, d = vw.calculate_sift_descriptors(image, STEP_SIZE, CELL_SIZE, norm_threshold=0.00)

    print("Quantizing. This may take a while...")
    l = quantization.quantize_descriptors(d, visualwords)

    Save k(eypoints) and l(abels) like shown below.
    """

    # calculate codebook and keypoints/labels for each page
    images = [mimg.imread("{}/{}.png".format(PAGE_FOLDER, d)) for d in FILENAMES]
    visualwords, keypoints, labels, page_sizes = vw.calculate_visual_words_from_images(images,
                                                                                       N_CENTROIDS,
                                                                                       STEP_SIZE,
                                                                                       CELL_SIZE,
                                                                                       norm_threshold=0.00)
    print("Descriptors per page:")
    print(page_sizes)

    # offsets for labels that belong to each page
    label_offsets = [0] + [a for a in mathutils.accumulate(page_sizes)]

    # save visualwords
    print("Writing visualwords with shape {}".format(visualwords.shape))
    np.save("vw_{}_{}x{}.npy".format(N_CENTROIDS, STEP_SIZE, CELL_SIZE), visualwords)

    # save labels for each page
    for idx, name in enumerate(FILENAMES):
        print("writing {}.npy".format(name))
        with open("{}.npy".format(name), "w") as outfile:
            print("... indices {} to {}".format(label_offsets[idx], label_offsets[idx+1]))
            np.save(outfile, keypoints[label_offsets[idx]:label_offsets[idx+1]])
            np.save(outfile, labels[label_offsets[idx]:label_offsets[idx+1]])

if __name__ == '__main__':
    log.start("extract_gw_codebook.py")
    extract_vw()
    log.end()
