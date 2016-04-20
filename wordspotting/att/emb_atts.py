# -*- coding: utf-8 -*-
"""
@author: Christian Wieprecht
@license: Apache License, Version 2.0

Building Attribute SVMS as described in
"Word Spotting and Recognition with Embedded Attributes" by Almazan et al.
"""
import numpy as np

import platt
import tools.logging as log
import svm


class AttributesSVMGenerator():
    def __init__(self):
        self.attribute_svms = None
        self.platts = False
        self.transform = None
        self.train_scores = None

    def fit(self, X, Y, platts=False):
        self.attribute_svms = svm.learn_embedded_attributes(X, Y)
        self.train_scores = svm.classify_embedded_attributes(X, self.attribute_svms)
        if platts:
            self.platts = True
            self.__learn_sigmoid_params(X)

    def __learn_sigmoid_params(self, data_mat):
        log.d("Platt's scaling (visual feature SVMs)...")
        class_labels = svm.predict_embedded_attributes_labels(data_mat, self.attribute_svms)
        self.sigmoid_params = platt.learn_platts_scaling_params(self.train_scores, class_labels)
        del class_labels

    def score(self, X):
        scores = svm.classify_embedded_attributes(X, self.attribute_svms)
        if self.platts:
            return platt.perform_platts_scaling(scores, self.sigmoid_params).T
        if self.transform is not None:
            # there is no nice way to test ndarray against None...
            if not np.equal(self.transform, None):
                return np.dot(scores.T, self.transform)
        return scores.T

    def save_to_file(self, path):
        log.d("Saving svms to {}".format(path))
        f = file(path, "wb")
        np.save(f, self.attribute_svms)
        np.save(f, self.platts)
        np.save(f, self.transform)
        if self.platts:
            np.save(f, self.sigmoid_params)
        f.close()

    def load_from_file(self, path):
        f = file(path, "rb")
        self.attribute_svms = np.load(f)
        self.platts = np.load(f)
        self.transform = np.load(f)
        if self.platts:
            self.sigmoid_params = np.load(f)
        f.close()
