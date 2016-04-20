# -*- coding: utf-8 -*-
"""
@author: Christian Wieprecht
@license: Apache License, Version 2.0
"""
import numpy as np


def read_trajectory_from_file(file):
    """
    Reads points of an online trajectory from a textfile where
    each line is formatted as "x y penup". All entries have to be integers.
    penup is 0/1 depending on the state of the pen. An optional annotation of
    the presented word can be given on the first line

    @param file: Textfile containing points of a trajectory.
    @return: Numpy array with columns x, y and penup, annotation or None
    """
    points = []
    annotation = None
    with open(file, "r") as traj_file:
        for line in traj_file:
            parts = line.split(" ")
            if len(parts) != 3:
                annotation = line.strip()
                continue
            points.append([int(p.strip()) for p in parts])
    return np.array(points), annotation


def read_traj_clusters(filename):
    """
    Parses a textfile of online clusters, where each line contains
    the space-separated values of one cluster-center.
    """
    read_line = lambda s: [float(p) for p in s.split(" ")]
    with open(filename, "r") as infile:
        traj_clusters = np.array([read_line(line) for line in infile])
    return traj_clusters


def read_traj_keypoints(filename):
    points = []
    with open(filename, "r") as keypoint_file:
        for line in keypoint_file:
            parts = line.split(" ")
            points.append([int(p.strip()) for p in parts])
    return np.array(points)


def read_traj_labels(filename):
    labels = []
    with open(filename, "r") as keypoint_file:
        for line in keypoint_file:
            labels.append(int(line.strip()))
    return np.array(labels)