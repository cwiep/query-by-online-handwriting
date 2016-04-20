# -*- coding: utf-8 -*-
"""
@author: Christian Wieprecht
@license: Apache License, Version 2.0

Interface to handle online-handwritten queries as created with the demo application.
NOTE: Needs a lot of precalculated data, where the paths need to be set correctly below.
"""
import numpy as np

from wordspotting.traj import trajfeat, trajnorm, trajimport
from wordspotting.att import emb_atts
from wordspotting.text import transcription as tt
from wordspotting import quantization, spatialpyramid
from tools import evaluation
from tools import mathutils

NORM_STEPS = ["flip", "slope", "resample", "slant", "height"]
FEAT_STEPS = ["dir", "curv", "vic_aspect", "vic_curl", "vic_line", "vic_slope", "bitmap"]
SP_CONF    = [[9, 2], [3, 2]]

ROOT_FOLDER = "/home/cwieprec/Work/MA.demo"

GW_GROUNDTRUTH = "/home/cwieprec/Work/MA.data/gw_ls12/GT"
GW_DOCUMENTS  = ["2700270", "2710271", "2720272", "2730273", "2740274", "2750275", "2760276", "2770277",
                 "2780278", "2790279", "3000300", "3010301", "3020302", "3030303", "3040304", "3050305",
                 "3060306", "3070307", "3080308", "3090309"]
GW_WORDS = "{}/data/gw_words.txt".format(ROOT_FOLDER)

TRAJ_CLUSTERS = "{}/data/clusters_online_up_17feat.txt".format(ROOT_FOLDER)
TRAJ_SVMS = "{}/data/traj_svms_platts.npy".format(ROOT_FOLDER)
CORPUS_MATRIX = "{}/data/vis_att_mat_platts_gw_vw2048_sp_2x1_1x1.txt.gz".format(ROOT_FOLDER)


class QboWordspotting(object):
    def __init__(self):
        print("Loading online-handwriting feature clusters...")
        self.traj_clusters = trajimport.read_traj_clusters(TRAJ_CLUSTERS)
        print("Loading corpus mat...")
        self.corpus_mat = np.loadtxt(CORPUS_MATRIX)
        
        self.test_words = []
        with open(GW_WORDS, "r") as f:
            for w in f:
                self.test_words.append(w.strip())
        self.traj_svms = emb_atts.AttributesSVMGenerator()
        self.traj_svms.load_from_file(TRAJ_SVMS)
        self.word_index = self.__get_gw_trans_data_pages()
        self.document_sizes = self.__get_document_sizes()
        
    def retrieve_hits(self, query_traj, document_id=None, num_results=10):
        print("Trajectory with {} points".format(len(query_traj)))
        
        # calculate online-handwriting feature sequence
        traj_normed = trajnorm.normalize_trajectory(np.array(query_traj), NORM_STEPS)
        feat_vec_seq = trajfeat.calculate_feature_vector_sequence(traj_normed, FEAT_STEPS)
        
        # quantize feature vectors
        labels = quantization.quantize_descriptors(feat_vec_seq, self.traj_clusters)
        
        # calculate spatial pyramid
        desc = self.__calculate_spatial_pyramid(traj_normed, labels)
        
        # "input vector should be 1-D"...
        desc_ = np.zeros(shape=(1, len(desc)))
        desc_[0] = desc
        print("Final descriptor has shape " + str(desc_.shape))
        
        # classify phoc attributes
        feat_vec = self.traj_svms.score(desc_)

        # retrieve results
        if document_id is None:
            # support old demo tool that searches in all documents
            top_results = evaluation.get_top_results(self.corpus_mat, feat_vec, self.test_words, 10)
            for r in top_results:
                print(str(self.word_index[r[0]]) + ', score: ' + str(r[2]))
            return
        
        doc_index = GW_DOCUMENTS.index(document_id)

        # calculate region of corpus matrix for given document
        begin_index = self.document_sizes[doc_index]
        end_index = self.document_sizes[doc_index+1]

        top_results = evaluation.get_top_results(self.corpus_mat[begin_index:end_index], feat_vec, self.test_words[begin_index:end_index], num_results)

        # create retrieval matrix
        ret_mat = np.zeros((6, num_results))
        for i in range(num_results):
            # result format is (row index of corpus mat, word, score)
            result = top_results[i]
            word_object = self.trans_data_pages[doc_index][result[0]]
            # score, relevance, ul_x, ul_y, lr_x, lr_y
            ret_mat[:, i] = [result[2], 0, word_object.xstart, word_object.ystart, word_object.xend, word_object.yend]
        
        return ret_mat
            
    def __calculate_spatial_pyramid(self, traj_normed, labels):
        minx = min(traj_normed[:, 0])
        maxx = max(traj_normed[:, 0])
        miny = min(traj_normed[:, 1])
        maxy = max(traj_normed[:, 1])
        spatial_pyramid = spatialpyramid.SpatialPyramid(SP_CONF, len(self.traj_clusters))
        return spatial_pyramid.calculate_descriptor(traj_normed[:, :2], labels, (minx, miny), maxx-minx, maxy-miny)
        
    def __get_gw_trans_data_pages(self):
        self.trans_data_pages = tt.load_transcription_data(GW_GROUNDTRUTH, GW_DOCUMENTS)
        # flatten pages to allow for direct word indexing
        return [w for p in self.trans_data_pages for w in p]

    def __get_document_sizes(self):
        return [0] + list(mathutils.accumulate([len(p) for p in self.trans_data_pages]))
