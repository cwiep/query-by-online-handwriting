#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Christian Wieprecht
@license: Apache License, Version 2.0

Input: Keypoints and Labels for GW, GWO and Unipen data
Output: Feature vector matrix for a given dataset, n_samples x n_features,
        where n_features depends on size of spatial pyramid and codebook

Abbreviations:
noncls = number of online clusters
nviscs = number of visual clusters
osps = online spatial pyramids
vsps = visual spatial pyramids
"""
import os
import numpy as np

from wordspotting.bof import visimport
from wordspotting.bof.bagoffeatures import BofGenerator
from wordspotting.spatialpyramid import SpatialPyramid
import wordspotting.text.transcription as trans
from wordspotting.traj import trajimport
from wordspotting.traj.boof import BoofGenerator
import tools.logging as log


ROOT_FOLDER = "/home/chris/Work/MA.data"
OUT_FOLDER = "/home/chris/Work/MA.data/precomputed"


def precalculate_unipen(nonlcs, osps):
    unipen_folders = ["aeb", "asl", "ben", "cb", "ckb", "dlm", "etb", "gl", "ja", "jdc", "jhc", "jma", "kaj", "kew",
                      "ksc", "lac", "lcf", "mek", "mml", "mmm", "nco", "pm", "rn", "rv", "sbc", "scd", "sd", "sij",
                      "skh", "skw", "srs"]
    # we need to filter out the few word occurences in the chosen unipen subset that contain uppercase letters
    contains_uppercase = lambda t: any([c in t for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"])
    unipen_traj_keypoint_paths = ["{}/unipen/keypointss/sta0.hpb0-{}/{}".format(ROOT_FOLDER, upfol, t)
                        for upfol in unipen_folders
                        for t in sorted(os.listdir("{}/unipen/keypointss/sta0.hpb0-{}".format(ROOT_FOLDER, upfol)))
                        if not contains_uppercase(t)]

    for nonlc in nonlcs:
        unipen_traj_label_paths = [p.replace("keypointss", "labels/{}".format(nonlc)).replace("keypoints", "ow") for p in unipen_traj_keypoint_paths]
        unipen_traj_keypoint_data = [trajimport.read_traj_keypoints(p) for p in unipen_traj_keypoint_paths]
        unipen_traj_label_data = [trajimport.read_traj_labels(p) for p in unipen_traj_label_paths]

        for osp in osps:
            print("")
            log.d("[Loading Trainingdata]")

            traj_spatial_pyramid = SpatialPyramid(osp, nonlc)
            test_boof_generator = BoofGenerator(traj_spatial_pyramid)
            traj_feat_mat = test_boof_generator.build_feature_vectors_matrix(unipen_traj_keypoint_data, unipen_traj_label_data)

            outname = __unipen_param_config_string(nonlc, osp) + ".txt.gz"
            log.d("Saving {}".format(outname))
            np.savetxt("{}/{}".format(OUT_FOLDER, outname), traj_feat_mat)


def precalculate_gwo(nonlcs, osps):
    filenames = ['2700270', '2710271', '2720272', '2730273', '2740274',
                 '2750275', '2760276', '2770277', '2780278', '2790279',
                 '3000300', '3010301', '3020302', '3030303', '3040304',
                 '3050305', '3060306', '3070307', '3080308', '3090309']

    gwo_traj_folder = "{}/gw_online".format(ROOT_FOLDER)
    gwo_traj_keypoint_paths = []
    for folder in filenames:
        path = "{}/keypointss/{}".format(gwo_traj_folder, folder)
        traj_names = sorted(file for file in os.listdir(path) if file.endswith(".keypoints"))
        gwo_traj_keypoint_paths.extend(["{}/{}".format(path, traj) for traj in traj_names])

    for nonlc in nonlcs:
        gwo_traj_label_paths = [p.replace("keypointss", "labels/{}".format(nonlc)).replace("keypoints", "ow") for p in gwo_traj_keypoint_paths]
        gwo_traj_keypoint_data = [trajimport.read_traj_keypoints(p) for p in gwo_traj_keypoint_paths]
        gwo_traj_label_data = [trajimport.read_traj_labels(p) for p in gwo_traj_label_paths]

        for osp in osps:
            print("")
            log.d("[Loading Trainingdata]")

            traj_spatial_pyramid = SpatialPyramid(osp, nonlc)
            test_boof_generator = BoofGenerator(traj_spatial_pyramid)
            traj_feat_mat = test_boof_generator.build_feature_vectors_matrix(gwo_traj_keypoint_data, gwo_traj_label_data)

            outname = __gwo_param_config_string(nonlc, osp) + ".txt.gz"
            log.d("Saving {}".format(outname))
            np.savetxt("{}/{}".format(OUT_FOLDER, outname), traj_feat_mat)


def precalculate_vis(nviscs, vsps):
    step_size = 5

    # contains .npy files for each page
    train_page_data_folder = "{}/gw_ls12/page_data".format(ROOT_FOLDER)
    # contains word image annotations for each page
    train_page_trans_folder = "{}/gw_ls12/GT".format(ROOT_FOLDER)
    filenames = ['2700270', '2710271', '2720272', '2730273', '2740274',
                 '2750275', '2760276', '2770277', '2780278', '2790279',
                 '3000300', '3010301', '3020302', '3030303', '3040304',
                 '3050305', '3060306', '3070307', '3080308', '3090309']

    trans_data_pages = trans.load_transcription_data(train_page_trans_folder, filenames)

    for nvisc in nviscs:
        log.d("Loading data for {} training pages...".format(len(filenames)))
        page_data_files = ["{}/{}/{}.npy".format(train_page_data_folder, nvisc, f) for f in filenames]
        keypoint_data_pages, label_data_pages = visimport.load_page_data(page_data_files)

        for vsp in vsps:
            spatial_pyramid = SpatialPyramid(vsp, nvisc)
            vis_feat_mat = BofGenerator(spatial_pyramid).build_feature_vectors_matrix(trans_data_pages,
                                                                                      keypoint_data_pages,
                                                                                      label_data_pages,
                                                                                      step_size)

            outname = __gw_param_config_string(nvisc, vsp) + ".txt.gz"
            log.d("Saving {}".format(outname))
            np.savetxt("{}/{}".format(OUT_FOLDER, outname), vis_feat_mat)


def __gw_param_config_string(nvisc, vsp):
    result = "vis_feat_mat_gw_vw{}_sp".format(nvisc)
    return result + __spatial_pyramid_config_string(vsp)


def __gwo_param_config_string(nvisc, vsp):
    result = "traj_feat_mat_gwo_vw{}_sp".format(nvisc)
    return result + __spatial_pyramid_config_string(vsp)


def __unipen_param_config_string(nvisc, vsp):
    result = "traj_feat_mat_unipen_vw{}_sp".format(nvisc)
    return result + __spatial_pyramid_config_string(vsp)


def __spatial_pyramid_config_string(spatial_pyramid):
    result = ""
    for level in spatial_pyramid:
        result += "_{}x{}".format(level[0], level[1])
    return result


if __name__ == "__main__":
    #vsps = [[[2, 1], [1, 1]], [[3, 2], [2, 1]], [[6, 2], [2, 1]]]
    #nviscs = [2048, 4096, 8192]
    #precalculate_vis(nviscs, vsps)

    #osps = [[[3, 2], [2, 1]], [[6, 2], [2, 1]], [[9, 2], [3, 2]]]
    #nonlcs = [128, 256, 512]
    #precalculate_gwo(nonlcs, osps)

    osps = [[[3, 2], [2, 1]], [[6, 2], [2, 1]], [[9, 2], [3, 2]]]
    nonlcs = [128, 256, 512]
    precalculate_unipen(nonlcs, osps)
