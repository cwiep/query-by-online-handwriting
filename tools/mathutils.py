# -*- coding: utf-8 -*-
"""
@author: Christian Wieprecht
@license: Apache License, Version 2.0
"""
import operator
import numpy as np


def accumulate(iterable, func=operator.add):
    """
    Return running totals.
    Equivalent to itertools.accumulate which is not present in python 2

    accumulate([1,2,3,4,5]) --> 1 3 6 10 15
    accumulate([1,2,3,4,5], operator.mul) --> 1 2 6 24 120
    """
    it = iter(iterable)
    total = next(it)
    yield total
    for element in it:
        total = func(total, element)
        yield total


def normalize(v):
    norm=np.linalg.norm(v)
    if norm==0:
       return v
    return v/norm


def power_normalize(v, alpha=0.5):
    fun = lambda x: abs(x)**alpha if x >= 0 else -abs(x)**alpha
    vec_fun = np.vectorize(fun)
    return vec_fun(v)


def remove_duplicates(seq):
    seen = set()
    seen_add = seen.add
    return [ x for x in seq if not (x in seen or seen_add(x))]