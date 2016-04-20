"""
Copyright notice for original file:

Copyright (c) 2000-2014 Chih-Chung Chang and Chih-Jen Lin
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither name of copyright holders nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.


THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import numpy as np
from math import log, exp

# Source: http://www.work.caltech.edu/~htlin/program/libsvm/

# H.-T. Lin, C.-J. Lin, and R. C. Weng. 
# A Note on Platt's Probabilistic Outputs for Support Vector Machines, 2003. 


def _sigmoid_train(deci, label, prior1=None, prior0=None):
    """
    decision_values, real_labels{1,-1}, #positive_instances, #negative_instances
    @return: [A,B] that minimize sigmoid likilihood
    """
    # Count prior0 and prior1 if needed
    if prior1 == None or prior0 == None:
        prior1, prior0 = 0, 0
        for i in range(len(label)):
            if label[i] > 0:
                prior1 += 1
            else:
                prior0 += 1

    #Parameter Setting
    maxiter = 100  #Maximum number of iterations
    minstep = 1e-10  #Minimum step taken in line search
    sigma = 1e-12  #For numerically strict PD of Hessian
    eps = 1e-5

    #Construct Target Support
    hiTarget = (prior1 + 1.0) / (prior1 + 2.0)
    loTarget = 1 / (prior0 + 2.0)
    length = prior1 + prior0
    t = []

    for i in range(length):
        if label[i] > 0:
            t.append(hiTarget)
        else:
            t.append(loTarget)

    #Initial Point and Initial Fun Value
    A, B = 0.0, log((prior0 + 1.0) / (prior1 + 1.0))
    fval = 0.0

    for i in range(length):
        fApB = deci[i] * A + B
        if fApB >= 0:
            fval += t[i] * fApB + log(1 + exp(-fApB))
        else:
            fval += (t[i] - 1) * fApB + log(1 + exp(fApB))

    for it in range(maxiter):
        #Update Gradient and Hessian (use H' = H + sigma I)
        h11 = h22 = sigma  #Numerically ensures strict PD
        h21 = g1 = g2 = 0.0
        for i in range(length):
            fApB = deci[i] * A + B
            if (fApB >= 0):
                p = exp(-fApB) / (1.0 + exp(-fApB))
                q = 1.0 / (1.0 + exp(-fApB))
            else:
                p = 1.0 / (1.0 + exp(fApB))
                q = exp(fApB) / (1.0 + exp(fApB))
            d2 = p * q
            h11 += deci[i] * deci[i] * d2
            h22 += d2
            h21 += deci[i] * d2
            d1 = t[i] - p
            g1 += deci[i] * d1
            g2 += d1

        #Stopping Criteria
        if abs(g1) < eps and abs(g2) < eps:
            break

        #Finding Newton direction: -inv(H') * g
        det = h11 * h22 - h21 * h21
        dA = -(h22 * g1 - h21 * g2) / det
        dB = -(-h21 * g1 + h11 * g2) / det
        gd = g1 * dA + g2 * dB

        #Line Search
        stepsize = 1
        while stepsize >= minstep:
            newA = A + stepsize * dA
            newB = B + stepsize * dB

            #New function value
            newf = 0.0
            for i in range(length):
                fApB = deci[i] * newA + newB
                if fApB >= 0:
                    newf += t[i] * fApB + log(1 + exp(-fApB))
                else:
                    newf += (t[i] - 1) * fApB + log(1 + exp(fApB))

            #Check sufficient decrease
            if newf < fval + 0.0001 * stepsize * gd:
                A, B, fval = newA, newB, newf
                break
            else:
                stepsize = stepsize / 2.0

        if stepsize < minstep:
            print "line search fails", A, B, g1, g2, dA, dB, gd
            return [A, B]

    if it >= maxiter - 1:
        print "reaching maximal iterations", g1, g2
    return [A, B]


def _sigmoid_predict(svm_score, AB):
    """
    Returns prediction probability for SVM-score.
    """
    A, B = AB
    fApB = svm_score * A + B
    if (fApB >= 0):
        return exp(-fApB) / (1.0 + exp(-fApB))
    else:
        return 1.0 / (1 + exp(fApB))


"""
Helper functions

@author: Christian Wieprecht
@license: Apache License, Version 2.0
"""


def learn_platts_scaling_params(svm_scores, class_labels):
    """
    Learns one sigmoid function for each attribute-SVM.

    @param svm_scores: Matrix where each column contains SVM-scores for one feature vector.
    @param class_labels: Matrix where each column class labels for one feature vector.
    """
    return np.array([_sigmoid_train(svm_scores[idx], class_labels[idx]) for idx in range(svm_scores.shape[0])])


def perform_platts_scaling(svm_scores, ABs):
    """
    Applies sigmoid functions ABs to SVM-scores.

    @param svm_scores: Matrix where each column contains SVM-scores for one feature vector.
    @param ABs: Array containing sigmoid parameters [A, B] in each row.
    @return: Matrix where each column contains scaled feature vector.
    """
    return np.array([[_sigmoid_predict(score, ABs[idx]) for score in svm_scores[idx]] for idx in range(svm_scores.shape[0])])
