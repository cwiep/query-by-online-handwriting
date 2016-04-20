#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Christian Wieprecht
@license: Apache License, Version 2.0

This module contains the code used for the known writer experiments using the LSA method coducted in the paper
"Word Spotting in Historical Documents with Online-Handwritten Queries".
"""
import numpy as np
from sklearn.decomposition import TruncatedSVD

import tools.logging as log
from tools import mathutils
from tools import evaluation
import wordspotting.text.transcription as tt


def __gw_param_config_string(nvisc, vsp):
    result = "vis_feat_mat_gw_vw{}_sp".format(nvisc)
    return result + __spatial_pyramid_config_string(vsp)


def __gwo_param_config_string(nvisc, vsp):
    result = "traj_feat_mat_gwo_vw{}_sp".format(nvisc)
    return result + __spatial_pyramid_config_string(vsp)

def __spatial_pyramid_config_string(spatial_pyramid):
    result = ""
    for level in spatial_pyramid:
        result += "_{}x{}".format(level[0], level[1])
    return result

def __get_gw_trans_data_pages(ma_data):
    gw_page_trans_folder = "{}/gw_ls12/GT".format(ma_data)
    gw_filenames = ['2700270', '2710271', '2720272', '2730273', '2740274',
                    '2750275', '2760276', '2770277', '2780278', '2790279',
                    '3000300', '3010301', '3020302', '3030303', '3040304',
                    '3050305', '3060306', '3070307', '3080308', '3090309']
    return tt.load_transcription_data(gw_page_trans_folder, gw_filenames)


def run_experiment():
    ma_data = "/home/cwieprec/Work/MA.data"
    precomputed = "{}/precomputed".format(ma_data)
    xval_num_folds = 4

    # list of dictionaries that contain all parameter combinations that will be run
    param_combs = [{"vw": 2048, "vsp": [[3, 2], [2, 1]], "ow": 512, "osp": [[9, 2], [3, 2]]}]

    # load annotation data
    gw_trans_data_pages = __get_gw_trans_data_pages(ma_data)

    # all word annotations in dataset
    gw_words = np.array([w.word for p in gw_trans_data_pages for w in p])

    num_pages = len(gw_trans_data_pages)
    accum_page_sizes = [0] + list(mathutils.accumulate([len(p) for p in gw_trans_data_pages]))
    num_samples = accum_page_sizes[-1]
    fold_size = num_pages / xval_num_folds

    best_map = 0.0
    best_param_comb = {}

    for param_comb in param_combs:
        nonlc = param_comb["ow"]
        osp = param_comb["osp"]
        nvisc = param_comb["vw"]
        vsp = param_comb["vsp"]

        log.d("Loading and concatenating feature matrices...")
        gwo_traj_feat_mat = np.loadtxt("{}/{}.txt.gz".format(precomputed,__gwo_param_config_string(nonlc, osp)))
        vis_feat_mat = np.loadtxt("{}/{}.txt.gz".format(precomputed, __gw_param_config_string(nvisc, vsp)))

        # build shared feature matrix
        feat_mat = np.concatenate((vis_feat_mat, gwo_traj_feat_mat), axis=1)

        stats = []
        n_visual_features = vis_feat_mat.shape[1]
        n_online_features = gwo_traj_feat_mat.shape[1]
        n_features = n_visual_features + n_online_features

        for test_fold in range(xval_num_folds):
            test_pages_indices = range(test_fold * fold_size, test_fold * fold_size + fold_size)
            test_indices = [wi for pi in test_pages_indices for wi in range(accum_page_sizes[pi], accum_page_sizes[pi+1])]
            train_indices = [i for i in range(num_samples) if i not in test_indices]

            log.d("Calculating topic space with {} samples...".format(len(train_indices)))

            svd = TruncatedSVD(n_components=256)
            svd.fit(feat_mat[train_indices])

            """
            Evaluation
            """

            # in contrast to qbs-lsa we want to query all trajectories here
            corpus_words = gw_words[test_indices]
            query_words = corpus_words

            # for test corpus only take visual features
            corpus_mat = np.zeros(shape=(len(test_indices), n_features))
            corpus_mat[:, :n_visual_features] = feat_mat[test_indices, :n_visual_features]

            corpus_mat = svd.transform(corpus_mat)

            # for test queries only take trajectory features
            query_mat = np.zeros(shape=(len(test_indices), n_features))
            query_mat[:, n_visual_features:] = feat_mat[test_indices, n_visual_features:]
            query_mat = svd.transform(query_mat)

            print("")
            stats.append(evaluation.run_evaluation(query_mat, query_words, corpus_mat, corpus_words))
            print("")

        log.d("online centroids: {}, online spatial pyramid: {}, visual words: {}, visual spatial pyramid: {}".format(nonlc, osp, nvisc, vsp))
        evaluation.log_xval_stats(stats)

        test_stats = evaluation.get_xval_stats(stats)
        if test_stats["mAP"] > best_map:
            best_map = test_stats["mAP"]
            best_param_comb = param_comb

    log.d("Best parameter configuration is")
    log.d(best_param_comb)


if __name__ == '__main__':
    log.start("run_lsa_qbo_known_writer.py")
    run_experiment()
    log.end()








