# -*- coding: utf-8 -*-
"""
@author: Christian Wieprecht
@license: Apache License, Version 2.0

Learning embedded attributes as described in
"Word Spotting and Recognition with Embedded Attributes" by Almazan et al.
"""

import numpy as np
import sklearn
from sklearn import svm

import tools.logging as log
from tools.logging import update_progress


def classify_embedded_attributes(data_mat, svms):
    """
    Calculate SVM-scores for each feature vector (=row) in data_mat.

    @return: Matrix with each column representing PHOC-transformation of one feature vector.
    """
    num_attributes = len(svms)
    num_examples = data_mat.shape[0]
    A = np.zeros(shape=(num_attributes, num_examples))
    log.d("Classifying {} examples...".format(num_examples))
    for att_idx, svm in enumerate(svms):
        update_progress(att_idx + 1, num_attributes)
        if svm is not None:
            if sklearn.__version__ == '0.14.1':
                A[att_idx] = svm.decision_function(data_mat)
            else:
                # the return format of this function was changed in 0.15...
                A[att_idx] = svm.decision_function(data_mat).T
    print("")
    return A


def learn_embedded_attributes(data_mat, phoc_mat):
    """
    Learns one SVM for each PHOC-attribute.

    @param data_mat: Each row is a feature vector.
    @param phoc_mat: Each row is a phoc vector.
    @return: One SVM per PHOC attribute.
    """
    num_attributes = phoc_mat.shape[1]
    svms = np.empty(num_attributes, dtype=object)
    invalid = 0
    log.d("Training SVMs for {} attributes...".format(num_attributes))
    for att in range(num_attributes):
        log.update_progress(att + 1, num_attributes)
        labels = phoc_mat[:, att]
        # if we have only one class (either 1 or 0) we can't train a svm
        if sum(labels) == 0 or sum(labels) == num_attributes:
            svms[att] = None
            invalid += 1
            continue
        clf = svm.LinearSVC()
        clf.fit(data_mat, labels)
        svms[att] = clf
    print("")
    log.d("{} invalid attributes".format(invalid))
    return svms


def predict_embedded_attributes_labels(data_mat, svms):
    """
    Calculate class label predictions for each feature vector (=row) in data_mat.

    @return: Matrix with each column containing class labels for one feature vector.
    """
    num_attributes = len(svms)
    num_examples = data_mat.shape[0]
    A = np.zeros(shape=(num_attributes, num_examples))
    log.d("Classifying {} examples...".format(num_examples))
    for att_idx, svm in enumerate(svms):
        log.update_progress(att_idx + 1, num_attributes)
        if svm is not None:
            if sklearn.__version__ == '0.14.1':
                A[att_idx] = svm.predict(data_mat)
            else:
                # the return format of this function was changed in 0.15...
                A[att_idx] = svm.predict(data_mat).T
    print("")
    return A