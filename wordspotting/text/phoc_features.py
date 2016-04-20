# -*- coding: utf-8 -*-
"""
@author: Christian Wieprecht
@license: Apache License, Version 2.0

Building PHOC feature vectors as described in
"Word Spotting and Recognition with Embedded Attributes" by Almazan et al.
"""
import numpy as np

import texttools
from tools import logging as log


class PhocFeatureGenerator():
    def __init__(self, trans_data_pages):
        self.trans_data_pages = trans_data_pages
        self.num_ngrams = 50
        self.ngram_size = 2
        self.ngram_dict = self._build_n_gram_dict()
        self.feat_vec_size = 604

    def _build_n_gram_dict(self):
        """
        Calculates the 50 (default) most common bigrams (default) from a
        list of pages, where each page is a list of WordData objects.
        """
        words = [wd.word for p in self.trans_data_pages for wd in p]
        ngrams = {}
        for w in words:
            w_ngrams = texttools.get_n_grams(w, self.ngram_size)
            for ng in w_ngrams:
                ngrams[ng] = ngrams.get(ng, 0) + 1
        sorted_list = sorted(ngrams.items(), key=lambda x: x[1], reverse=True)
        return {k: i for i, (k, v) in enumerate(sorted_list[:self.num_ngrams])}

    def build_textual_feature_vectors_matrix(self, trans_data_pages=None):
        """
        Builds matrix containing one row per word and one column per feature.
        """
        if trans_data_pages == None:
            pages = self.trans_data_pages
        else:
            pages = trans_data_pages
        num_rows = sum([len(t) for t in pages])
        text_feat_mat = np.zeros(shape=(num_rows, self.feat_vec_size))
        i = 0
        for trans_page in pages:
            for word_data in trans_page:
                text_feat_mat[i] = self.build_textual_feature_vector(word_data.word)
                i += 1
        log.d("PHOC feature-vector-matrix has shape {}".format(text_feat_mat.shape))
        return text_feat_mat

    def build_textual_feature_vector(self, word):
        """
        Calculate Pyramidal Histogram of Characters (PHOC) descriptor (See Almazan2014) for "word".
        """
        occupancy = lambda k, n: [float(k) / n, float(k+1) / n]
        overlap = lambda a, b: [max(a[0], b[0]), min(a[1], b[1])]
        size = lambda region: region[1] - region[0]
        n = len(word)
        char_indices = {d: i for i, d in enumerate("abcdefghijklmnopqrstuvwxyz0123456789")}
        levels = [2, 3, 4, 5]
        feature_vector = np.zeros(shape=(sum(levels * len(char_indices)) + 100))
        for index, char in enumerate(word):
            char_occ = occupancy(index, n)
            char_index = char_indices[char]
            for lev_index, level in enumerate(levels):
                for region in range(level):
                    region_occ = occupancy(region, level)
                    if size(overlap(char_occ, region_occ)) / size(char_occ) >= 0.5:
                        feat_vec_index = sum([l for l in levels if l < level]) * 36 + region * 36 + char_index
                        feature_vector[feat_vec_index] = 1
        # add 50 most common bigrams at level 2
        # an ngram is characterized by its first character's index
        ngram_features = np.zeros(100)
        ngram_occupancy = lambda k, n: [float(k) / n, float(k+2) / n]
        for i in range(n-1):
            ngram = word[i:i+2]
            if self.ngram_dict.get(ngram, 0) == 0:
                continue
            occ = ngram_occupancy(i, n)
            for region in range(2):
                region_occ = occupancy(region, 2)
                overlap_size = size(overlap(occ, region_occ)) / size(occ)
                if overlap_size >= 0.5:
                    ngram_features[region * 50 + self.ngram_dict[ngram]] = 1
        feature_vector[-100:] = ngram_features
        return feature_vector
