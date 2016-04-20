#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Christian Wieprecht
@license: Apache License, Version 2.0

Load all UNIPEN trajectories, run normalization and extract a feature vector sequence
for all of them. The feature vectors from all sequences are clustered into a
codebook. Keypoints and labels are saved for each input trajectory file.
"""

import os
import numpy as np

from tools import mathutils
import tools.logging as log
from wordspotting import clustering
import wordspotting.traj.trajfeat as feat
import wordspotting.traj.trajnorm as norm
from wordspotting.traj.trajimport import read_trajectory_from_file

NORM_ARGS = ["flip", "slope", "resample", "slant", "height", "origin"]
FEAT_ARGS = ["dir", "curv", "vic_aspect", "vic_curl", "vic_line", "vic_slope", "bitmap"]
N_CENTROIDS = 256

ROOT_FOLDER = "/home/cwieprec/Work/MA.data"
RAW_TRAJ_FOLDER = "{}/unipen/raw".format(ROOT_FOLDER)


def extract_ow():
    filenames = ["{}/{}/{}".format(RAW_TRAJ_FOLDER, dir, f)
                 for dir in os.listdir(RAW_TRAJ_FOLDER)
                 if os.path.isdir("{}/{}".format(RAW_TRAJ_FOLDER, dir))
                 for f in os.listdir("{}/{}".format(RAW_TRAJ_FOLDER, dir))
                 if f.endswith(".txt")]

    log.d("Loading trajectories and calculating feature vectors...")
    feature_vectors = []
    word_sizes = []
    points_normed = []
    for i, f in enumerate(filenames):
        traj, _ = read_trajectory_from_file(f)
        traj_normed = norm.normalize_trajectory(traj, NORM_ARGS)
        write_keypoints(traj_normed, f, ROOT_FOLDER)
        feat_vec = feat.calculate_feature_vector_sequence(traj_normed, FEAT_ARGS)
        feature_vectors.extend(feat_vec)
        word_sizes.append(len(traj_normed))
        points_normed.extend([[int(n) for n in p] for p in traj_normed])
        log.update_progress(i+1, len(filenames))
    print("")
    log.d("Accumulated {} feature vectors.".format(len(feature_vectors)))
    label_offsets = [0] + list(mathutils.accumulate(word_sizes))

    # cluster feature vectors
    log.d("Using Lloyd's algorithm to find {} clusters...".format(N_CENTROIDS))
    clusters, labels = clustering.cluster(np.array(feature_vectors), N_CENTROIDS)

    # create output dir
    try:
        os.makedirs("{}/unipen/clusters/{}".format(ROOT_FOLDER, N_CENTROIDS))
    except:
        pass

    # write codebook
    with open("{}/unipen/clusters/{}/clusters_online_up_17feat.txt".format(ROOT_FOLDER, N_CENTROIDS), "w") as outfile:
        log.d("Writing {} clusters...".format(clusters.shape[0]))
        for c in clusters:
            outfile.write(' '.join([str(feature) for feature in c]) + '\n')

    # save labels for each trajectory
    for idx, name in enumerate(filenames):
        out_writer_dir, fname = keypoints_outfile_for_inpath(name, N_CENTROIDS, ROOT_FOLDER)
        log.d("Writing {}.ow".format(fname))
        with open("{}/{}.ow".format(out_writer_dir, fname), "w") as outfile:
            start_label_idx = label_offsets[idx]
            end_label_idx = label_offsets[idx+1]
            print("... indices {} to {}".format(start_label_idx, end_label_idx))
            current_labels = labels[start_label_idx:end_label_idx]
            outfile.write('\n'.join([str(item) for item in current_labels]))


def write_keypoints(traj_normed, inpath, ma_data):
    fname = os.path.splitext(os.path.basename(inpath))[0]
    writer_dir = inpath.split("/")[-2]
    out_writer_dir = "{}/unipen/keypoints/{}".format(ma_data, writer_dir)
    try:
        os.makedirs(out_writer_dir)
    except:
        pass
    with open("{}/{}.keypoints".format(out_writer_dir, fname), "w") as outfile:
        for point in traj_normed:
            outfile.write("{} {}\n".format(int(point[0]), int(point[1])))


def keypoints_outfile_for_inpath(inpath, n_centroids, ma_data):
    fname = os.path.splitext(os.path.basename(inpath))[0]
    writer_dir = inpath.split("/")[-2]
    out_writer_dir = "{}/unipen/labels/{}/{}".format(ma_data, n_centroids, writer_dir)
    try:
        os.makedirs(out_writer_dir)
    except:
        pass
    return out_writer_dir, fname

if __name__ == '__main__':
    log.start("extract_unipen_codebook.py")
    extract_ow()
    log.end()
