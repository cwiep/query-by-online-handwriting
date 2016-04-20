#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Christian Wieprecht
@license: Apache License, Version 2.0

This module contains the code used for the unknown writer experiments coducted in the paper
"Word Spotting in Historical Documents with Online-Handwritten Queries".
"""
import os
import numpy as np

import tools.logging as log
from tools import mathutils
import tools.filesystem as filesys
from tools import evaluation
import wordspotting.text.transcription as tt
from wordspotting.att.emb_atts import AttributesSVMGenerator
from wordspotting.text.phoc_features import PhocFeatureGenerator


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


def __get_unipen_trans_data_pages(ma_data):
    unipen_folders = ["aeb", "asl", "ben", "cb", "ckb", "dlm", "etb", "gl", "ja", "jdc", "jhc", "jma", "kaj", "kew",
                      "ksc", "lac", "lcf", "mek", "mml", "mmm", "nco", "pm", "rn", "rv", "sbc", "scd", "sd", "sij",
                      "skh", "skw", "srs"]
    contains_uppercase = lambda t: any([c in t for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"])
    unipen_traj_keypoint_paths = ["{}/unipen/keypointss/sta0.hpb0-{}/{}".format(ma_data, upfol, t)
                                  for upfol in unipen_folders
                                  for t in sorted(os.listdir("{}/unipen/keypointss/sta0.hpb0-{}".format(ma_data, upfol)))
                                  if not contains_uppercase(t)]
    # we use the keypoints file names (which include the word annotation) to create the transcription object
    return tt.create_fake_transcription_data([filesys.basename(t) for t in unipen_traj_keypoint_paths])


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
    param_combs = [{"vw": 2048, "vsp": [[2, 1], [1, 1]], "ow": 128, "osp": [[9, 2], [3, 2]]}]

    # load annotation data
    unipen_trans_data_pages = __get_unipen_trans_data_pages(ma_data)
    gw_trans_data_pages = __get_gw_trans_data_pages(ma_data)

    # find words, that are present in both gw and unipen datasets
    words = np.array([w.word for p in gw_trans_data_pages for w in p])
    unipen_words = np.array([w.word for p in unipen_trans_data_pages for w in p])
    shared_words = mathutils.remove_duplicates([w for w in unipen_words if w in words])
    log.d("{} unique words".format(len(shared_words)))

    # calculate indices for feature matrices to use only those shared words
    gw_indices = [i for i, w in enumerate(words) if w in shared_words]
    gw_words = np.array([w for w in words if w in shared_words])
    unipen_indices = [i for i, w in enumerate(unipen_words) if w in shared_words]

    log.d("Building textual descriptors...")
    phoc_feature_generator = PhocFeatureGenerator(gw_trans_data_pages)
    gw_text_feat_mat = phoc_feature_generator.build_textual_feature_vectors_matrix()[gw_indices]
    unipen_text_feat_mat = phoc_feature_generator.build_textual_feature_vectors_matrix(unipen_trans_data_pages)[unipen_indices]

    num_pages = len(gw_trans_data_pages)
    accum_page_sizes = [0] + list(mathutils.accumulate([len([w for w in p if w.word in shared_words]) for p in gw_trans_data_pages]))
    num_samples = accum_page_sizes[-1]
    fold_size = num_pages / xval_num_folds

    best_map = 0.0
    best_param_comb = {}

    traj_svm_cache = {}

    for param_comb in param_combs:
        nonlc = param_comb["ow"]
        osp = param_comb["osp"]
        nvisc = param_comb["vw"]
        vsp = param_comb["vsp"]

        unipen_traj_feat_mat = np.loadtxt("{}/{}.txt.gz".format(precomputed,__unipen_param_config_string(nonlc, osp)))[unipen_indices]
        gwo_traj_feat_mat = np.loadtxt("{}/{}.txt.gz".format(precomputed,__gwo_param_config_string(nonlc, osp)))[gw_indices]

        """
        Train/Load online-handwriting SVMs
        """
        # unipen is always used as train set as a whole
        svm_key = str(nonlc) + str(osp)
        if svm_key in traj_svm_cache:
            traj_svms = traj_svm_cache[svm_key]
        else:
            traj_svms = AttributesSVMGenerator()
            traj_svms.fit(unipen_traj_feat_mat, unipen_text_feat_mat, platts=False)
            traj_svm_cache[svm_key] = traj_svms

        vis_feat_mat = np.loadtxt("{}/{}.txt.gz".format(precomputed, __gw_param_config_string(nvisc, vsp)))[gw_indices]

        stats = []

        for test_fold in range(xval_num_folds):
            test_pages_indices = range(test_fold * fold_size, test_fold * fold_size + fold_size)
            test_indices = [wi for pi in test_pages_indices for wi in range(accum_page_sizes[pi], accum_page_sizes[pi+1])]
            train_indices = [i for i in range(num_samples) if i not in test_indices]

            """
            Train visual SVMs
            """
            svms = AttributesSVMGenerator()
            svms.fit(vis_feat_mat[train_indices], gw_text_feat_mat[train_indices])

            # optionally learn regression which will set transform matrices in the svm objects
            # learn_regression(svms, traj_svms)

            """
            Evaluation
            """
            corpus_words = gw_words[test_indices]
            query_words = corpus_words

            corpus_mat = svms.score(vis_feat_mat[test_indices])
            query_mat = traj_svms.score(gwo_traj_feat_mat[test_indices])

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
    log.start("run_att_qbo_unknown_writer.py")
    run_experiment()
    log.end()
