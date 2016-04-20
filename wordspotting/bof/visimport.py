# -*- coding: utf-8 -*-
"""
@author: Christian Wieprecht
@license: Apache License, Version 2.0

Importing precalculated keypoints and labels for George Washington Dataset.
"""
import numpy as np
import tools.logging as log


def load_page_data(filenames):
    """
    Loads keypoints and labels for each file in filenames.

    @param filenames: List containing page filenames WITHOUT file-extension.
    @return: Keypoints-list, labels-list.
    """
    keypoints = []
    labels = []
    for f in filenames:
        with open(f, "r") as infile:
            k = np.load(infile)
            l = np.load(infile)
            keypoints.append(np.array(k))
            labels.append(np.array(l))
            log.d("Read {} keypoints and labels for page '{}'".format(len(k), f))
    return keypoints, labels


def load_page_data_ls12(data_folder, filenames):
    """
    Loads keypoints and labels for each file in filenames. data_folder must contain three
    files per entry in 'filenames': <filename>.keypoints_x, <filename>.keypoints_y,
    <filename>.vw.

    @param data_folder: Folder containing text files for each page in filenames.
    @param filenames: List containing page filenames WITHOUT file-extension.
    @return: Keypoints-list, labels-list and page-sizes-list.
    """
    keypoints = []
    labels = []
    for f in filenames:
        x = []
        y = []
        with open("{}/{}.keypoints_x".format(data_folder, f), "r") as infile:
            x_list = infile.readlines()
            x.extend([int(float(n.strip())) for n in x_list])
        with open("{}/{}.keypoints_y".format(data_folder, f), "r") as infile:
            y_list = infile.readlines()
            y.extend([int(float(n.strip())) for n in y_list])
        keypoints.append(np.array([x, y]).T)
        with open("{}/{}.vw".format(data_folder, f), "r") as infile:
            vw_list = infile.readlines()
            labels.append([int(float(n.strip()))-1 for n in vw_list])
        log.d("Read {} keypoints and labels for page '{}'".format(len(y_list), f))
    return keypoints, labels