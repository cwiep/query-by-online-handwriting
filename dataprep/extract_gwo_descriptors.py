#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Christian Wieprecht
@license: Apache License, Version 2.0

Load all GWO trajectories, run normalization and extract a feature vector sequence
for all of them. Quantize by a given codebook and save keypoints/labels.
"""

import tools.logging as log
from tools import mathutils
import os
import wordspotting.traj.trajfeat as feat
import wordspotting.traj.trajnorm as norm
from wordspotting.traj.trajimport import read_trajectory_from_file
from wordspotting import quantization
from wordspotting.traj import trajimport

NORM_ARGS = ["flip", "slope", "resample", "slant", "height", "origin"]
FEAT_ARGS = ["dir", "curv", "vic_aspect", "vic_curl", "vic_line", "vic_slope", "bitmap"]
N_CENTROIDS = 256

ROOT_FOLDER = "/home/chris/Work/MA.data"
CODEBOOK_FILE = "{}/unipen/clusters/{}/clusters_online_up_17feat.txt".format(ROOT_FOLDER, N_CENTROIDS)

OUT_FOLDER = "/home/chris/Work/test"

def extract_ow():
    ma_data = "/home/chris/Work/MA.data"

    pages = ['2700270']#, '2710271', '2720272', '2730273', '2740274',
             #'2750275', '2760276', '2770277', '2780278', '2790279',
             #'3000300', '3010301', '3020302', '3030303', '3040304',
             #'3050305', '3060306', '3070307', '3080308', '3090309']
    train_traj_folder = "{}/gw_online".format(ma_data)
    filenames = []
    for folder in pages:
        path = "{}/{}".format(train_traj_folder, folder)
        traj_names = sorted(file for file in os.listdir(path) if file.endswith(".txt"))
        filenames.extend(["{}/{}".format(path, traj) for traj in traj_names])

    # load trajectories
    log.d("Loading trajectories and calculating feature vectors...")

    # cluster feature vectors
    word_sizes = []
    labels = []
    traj_clusters = trajimport.read_traj_clusters(CODEBOOK_FILE)
    for i, f in enumerate(filenames):
        traj, _ = read_trajectory_from_file(f)
        traj_normed = norm.normalize_trajectory(traj, NORM_ARGS)
        write_keypoints(traj_normed, f)
        feat_vec = feat.calculate_feature_vector_sequence(traj_normed, FEAT_ARGS)
        labels.extend(quantization.quantize_descriptors(feat_vec, traj_clusters))
        #feature_vectors.extend(feat_vec)
        word_sizes.append(len(traj_normed))
        #points_normed.extend([[int(n) for n in p] for p in traj_normed])
        log.update_progress(i+1, len(filenames))
    print("")
    label_offsets = [0] + list(mathutils.accumulate(word_sizes))

    # save labels for each trajectory
    for idx, name in enumerate(filenames):
        out_writer_dir, fname = keypoints_outfile_for_inpath(name, N_CENTROIDS)
        log.d("Writing {}.ow".format(fname))
        with open("{}/{}.ow".format(out_writer_dir, fname), "w") as outfile:
            start_label_idx = label_offsets[idx]
            end_label_idx = label_offsets[idx+1]
            print("... indices {} to {}".format(start_label_idx, end_label_idx))
            current_labels = labels[start_label_idx:end_label_idx]
            outfile.write('\n'.join([str(item) for item in current_labels]))


def write_keypoints(traj_normed, inpath):
    fname = os.path.splitext(os.path.basename(inpath))[0]
    writer_dir = inpath.split("/")[-2]
    out_writer_dir = "{}/gw_online/keypoints/{}".format(OUT_FOLDER, writer_dir)
    try:
        os.makedirs(out_writer_dir)
    except:
        pass
    with open("{}/{}.keypoints".format(out_writer_dir, fname), "w") as outfile:
        for point in traj_normed:
            outfile.write("{} {}\n".format(int(point[0]), int(point[1])))


def keypoints_outfile_for_inpath(inpath, n_centroids):
    fname = os.path.splitext(os.path.basename(inpath))[0]
    writer_dir = inpath.split("/")[-2]
    out_writer_dir = "{}/gw_online/labels/{}/{}".format(OUT_FOLDER, n_centroids, writer_dir)
    try:
        os.makedirs(out_writer_dir)
    except:
        pass
    return out_writer_dir, fname

if __name__ == '__main__':
    log.start("extract_gwo_descriptors.py")
    extract_ow()
    log.end()
