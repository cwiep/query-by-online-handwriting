# -*- coding: utf-8 -*-
"""
@author: Christian Wieprecht
@license: Apache License, Version 2.0
"""
import os


basename = lambda p: p.split("/")[-1].split(".")[0]
listdir_fullpath = lambda d: [os.path.join(d, f) for f in os.listdir(d)]
word_from_path = lambda p: basename(p).split("_")[1]