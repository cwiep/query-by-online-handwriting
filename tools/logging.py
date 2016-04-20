# -*- coding: utf-8 -*-
"""
@author: Christian Wieprecht
@license: Apache License, Version 2.0
"""
import sys
import datetime
import pprint


def start(module=""):
    ts = datetime.datetime.now().strftime("%A, %d. %b %Y, %H:%M:%S")
    m = "" if module == "" else "/// file: {} ".format(module)
    print("======= Starttime: {} {}=======".format(ts, m))


def end():
    ts = datetime.datetime.now().strftime("%A, %d. %b %Y, %H:%M:%S")
    print("======= Endtime: {} =======".format(ts))


def i(msg, location=""):
    logstring = "{0:8}  {1}".format("[info]", msg)
    print(logstring)


def d(msg, location=""):
    logstring = "{0:8}  {1}".format("[debug]", msg)
    print(logstring)


def w(msg, location=""):
    logstring = "{0:8}  {1}".format("[warn]", msg)
    print(logstring)


def e(msg, location=""):
    logstring = "{0:8}  {1}".format("[error]", msg)
    print(logstring)


def log_array(a):
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(a)


def get_timestamp():
    return datetime.datetime.now().strftime("%H%M%S")


def update_progress(progress, target):
    """
    Display progress of some kind.

    @param progress: Number of accomplished tasks.
    @param target: Complete number of tasks.
    """
    progress_percent = 100.0 * float(progress) / target
    sys.stdout.write("\rProgress: {:.2f}% ({}/{} tasks finished)".format(progress_percent, progress, target))
    sys.stdout.flush()