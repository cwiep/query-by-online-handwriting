# -*- coding: utf-8 -*-
"""
@author: Christian Wieprecht
@license: Apache License, Version 2.0

Calculating regression for svm scores as described in
"Word Spotting and Recognition with Embedded Attributes" by Almazan et al.
"""
import numpy as np
from scipy.linalg import eig


def learn_regression_for_svms(svm1, svm2):
    A = svm1.train_scores
    B = svm2.train_scores
    P = learn_regression(A, B)
    svm1.transform = P


def learn_regression(A, B):
    num_features = A.shape[0]
    I = np.eye(num_features, num_features)
    alpha = 0.5
    a = np.dot(A, A.T)
    b = a + alpha * I
    c = np.linalg.inv(b)
    d = np.dot(A, B.T)
    P = np.dot(c, d)
    return P


def learn_common_subspace_regression_for_svms(svm1, svm2):
    A = svm1.train_scores
    B = svm2.train_scores
    U, V = learn_common_subspace_regression(A, B)
    svm1.transform = U
    svm2.transform = V


def learn_common_subspace_regression(A, B):
    num_features = A.shape[0]
    I = np.eye(num_features, num_features)
    alpha = 0.5
    a = np.dot(A, B.T)
    b = np.linalg.inv(np.dot(B, B.T) + alpha * I)
    c = np.dot(B, A.T)
    left = np.dot(np.dot(a, b), c)
    right = np.dot(A, A.T) + alpha * I
    U = eig(left, right)[1]
    d = np.linalg.inv(np.dot(B, B.T) + alpha * I)
    e = np.dot(d, c)
    V = np.dot(e, U)
    return U, V