# -*- coding: utf-8 -*-
"""
@author: Christian Wieprecht
@license: Apache License, Version 2.0
"""
import numpy as np
from scipy.spatial import distance as scidist

from tools import logging as log


def run_evaluation(query_mat, query_words, corpus_mat, corpus_words, drop_first=False):
    """
    Using each feature vector (row) in query_mat once as query, run comparisons with all
    feature vectors (rows) in corpus_mat.
    """
    num_queries = len(query_mat)
    log.d("Running {} queries...".format(num_queries))
    recall = 0.0
    precision = 0.0
    avg_precision = 0.0
    up = log.update_progress
    gtr = get_top_results
    er = __evaluate_results
    invalid = 0
    for i, query_vec in enumerate(query_mat):
        word = query_words[i]
        top_results = gtr(corpus_mat, query_vec, corpus_words, corpus_mat.shape[0], drop_first=drop_first)
        query_occurences = sum([1 for w in corpus_words if w == word])
        if drop_first:
            query_occurences -= 1
        ap, rec, prec = er(top_results, word, query_occurences, logresults=False)
        if ap is None:
            # dont count invalid queries
            invalid += 1
        else:
            recall += rec
            precision += prec
            avg_precision += ap
        up(i+1, num_queries)
    print("")
    num_queries -= invalid
    if num_queries == 0:
        log.e("No valid queries")
        return [0, 0, 0, 0]
    recall /= num_queries
    precision /= num_queries
    avg_precision /= num_queries

    __log_results("Results for {} queries:".format(num_queries), recall, precision, avg_precision)

    return [num_queries, recall, precision, avg_precision]


def run_evaluation_with_invocab(query_mat, query_words, corpus_mat, corpus_words, train_vocab, drop_first=False):
    """
    Same as run_evaluation, but also saving separate statistics for in vocabulary queries.
    """
    num_queries = len(query_mat)
    log.d("Running {} queries...".format(num_queries))
    recall = 0.0
    precision = 0.0
    avg_precision = 0.0
    iv_num_queries = 0
    iv_recall = 0.0
    iv_precision = 0.0
    iv_avg_precision = 0.0
    up = log.update_progress
    gtr = get_top_results
    er = __evaluate_results
    invalid = 0
    for i, query_vec in enumerate(query_mat):
        word = query_words[i]
        in_vocab = word in train_vocab
        top_results = gtr(corpus_mat, query_vec, corpus_words, corpus_mat.shape[0], drop_first=drop_first)
        query_occurences = sum([1 for w in corpus_words if w == word])
        if drop_first:
            query_occurences -= 1
        ap, rec, prec = er(top_results, word, query_occurences, logresults=False)
        if ap is None:
            # dont count invalid queries
            if not in_vocab:
                invalid += 1
        else:
            if in_vocab:
                iv_recall += rec
                iv_precision += prec
                iv_avg_precision += ap
                iv_num_queries += 1
            recall += rec
            precision += prec
            avg_precision += ap
        up(i+1, num_queries)
    print("")

    num_queries = num_queries - invalid
    if num_queries > 0:
        recall /= num_queries
        precision /= num_queries
        avg_precision /= num_queries

        __log_results("Results for {} overall queries:".format(num_queries), recall, precision, avg_precision)
    else:
        log.e("No queries.")

    if iv_num_queries > 0:
        iv_recall /= iv_num_queries
        iv_precision /= iv_num_queries
        iv_avg_precision /= iv_num_queries

        __log_results("Results for {} in-vocabulary queries:".format(iv_num_queries), iv_recall, iv_precision, iv_avg_precision)
    else:
        log.e("No in vocabulary queries.")

    return [num_queries, recall, precision, avg_precision, iv_num_queries, iv_recall, iv_precision, iv_avg_precision]


def get_top_results(corpus_mat, query_feat_vec, words, num_results, drop_first=False):
    """
    Calculate cosine-distances from query_feat_vec to each row in feat_mat.

    @param corpus_mat: Matrix where each row contains feature vector.
    @param query_feat_vec: Feature vector of query.
    @param words: words[i] contains word for feat_mat[i]
    @param num_results: Only return top <num_results> results.
    @param drop_first: Ignore best result.
    @return: List of type (index, word, distance) sorted by ascending distance.
    """
    dists = np.empty(shape=(corpus_mat.shape[0]), dtype=object)
    for i, d in enumerate(corpus_mat):
        dists[i] = (i, words[i], scidist.cosine(d, query_feat_vec))
        # dists[i] = (words[i], scidist.euclidean(d, query_feat_vec))
    if drop_first:
        return sorted(dists, key=lambda x: x[2])[1:min(num_results + 1, corpus_mat.shape[0])]
    else:
        return sorted(dists, key=lambda x: x[2])[:min(num_results, corpus_mat.shape[0])]


def __evaluate_results(top_results, query_word, query_occurences, logresults=True):
    """
    Calculate key figures from query results.

    @param top_results: List containing best results sorted ascending by distance to query feature vector.
        Contains tuples with format (wordstring, distance to query).
    @param query_word: Word that was searched.
    @param logresults: If True, also print recall, precision and average precision.
    @return: Average Precision, Recall, Precision
    """
    if query_occurences == 0:
        if logresults:
            log.d("Queryword {} is not represented in searched text.".format(query_word))
        return None, None, None
    num_results = len(top_results)
    found = 0
    it_num = 1
    top = 0
    for _, word, dist in top_results:
        if word == query_word:
            found += 1
            top += float(found)/it_num
        it_num += 1
    ap = 100 * float(top) / found if found != 0 else 0.0
    recall = 100 * float(found) / query_occurences
    precision = 100 * float(found) / num_results
    if logresults:
        __log_results("Query: '{}'".format(query_word), recall, precision, ap)
    return ap, recall, precision


def log_xval_stats(stats):
    separate_in_vocab = len(stats[0]) > 4
    overall = [0, 0, 0, 0]
    for res in stats:
        print(res)
        overall[0] += res[0]
        for i in range(1, 4):
            # weighting individiual statistics by number of queries
            overall[i] += res[0] * res[i]
    __log_results("Results for {} overall queries:".format(overall[0]), overall[1] / overall[0], overall[2] / overall[0], overall[3] / overall[0])

    if separate_in_vocab:
        overall = [0, 0, 0, 0]
        for res in stats:
            overall[0] += res[4]
            for i in range(1, 4):
                # weighting individiual statistics by number of queries
                overall[i] += res[4] * res[4+i]
        __log_results("Results for {} in-vocabulary queries:".format(overall[0]), overall[1] / overall[0], overall[2] / overall[0], overall[3] / overall[0])


def get_xval_stats(stats):
    overall = [0, 0, 0, 0]
    for res in stats:
        overall[0] += res[0]
        for i in range(1, 4):
            # weighting individiual statistics by number of queries
            overall[i] += res[0] * res[i]
    return {"queries": overall[0],
            "mR": overall[1] / overall[0],
            "mP": overall[2] / overall[0],
            "mAP": overall[3] / overall[0]}


def __log_results(header, mr, mp, map):
    print("")
    log.d(header)
    log.d("mR  {0:.2f}%".format(mr))
    log.d("mP  {0:.2f}%".format(mp))
    log.d("mAP {0:.2f}%".format(map))