# -*- coding: utf-8 -*-
"""
@author: Christian Wieprecht
@license: Apache License, Version 2.0

Building textual feature vectors as described in
"Integrating visual and textual cues for query-by-string word spotting" by Aldavert et al.
"""
from tools import mathutils
from wordspotting.text import texttools
import numpy as np
from tools import logging as log


class NgramFeatureGenerator():
    def __init__(self, trans_data_pages):
        self.trans_data_pages = trans_data_pages
        self.ngram_sizes = [1, 2, 3]
        self.codebook = self.build_n_gram_codebook()
        self.feat_vec_size = len(self.codebook)

    def build_n_gram_codebook(self):
        """
        Counts all uni-, bi- and trigrams that occur in training data pages.
        Underrepresented ngrams are discarded.
        """
        count = {}
        for page in self.trans_data_pages:
            for word in page:
                w = word.word
                grams = [ng for i in self.ngram_sizes for ng in texttools.get_n_grams(w, i)]
                for gram in grams:
                    count[gram] = count.get(gram, 0) + 1
        return {key: index for index, key in enumerate(count.keys())}

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
        log.d("n-gram feature-vector-matrix has shape {}".format(text_feat_mat.shape))
        return text_feat_mat

    def build_textual_feature_vector(self, word):
        """
        Creates a len(codebook)-dimensional feature vector of the given word containing
        uni-, bi- and three-gram information.
        """
        grams = [ng for i in self.ngram_sizes for ng in texttools.get_n_grams(word, i)]
        textual_descriptor = np.zeros(shape=(len(self.codebook)))
        for gram in grams:
            index = self.codebook.get(gram, None)
            if index is not None:
                textual_descriptor[index] += 1
        return mathutils.normalize(textual_descriptor)