# -*- coding: utf-8 -*-
"""
@author: Christian Wieprecht
@license: Apache License, Version 2.0
"""
import numpy as np
import io

import matplotlib.pyplot as plt
import matplotlib.image as mimg


def render_trajectory_to_png(traj, name, outfolder, invert_y=False):
    min_x = min(traj[:, 0]) - 5
    min_y = min(traj[:, 1]) - 5
    max_x = max(traj[:, 0]) + 5
    max_y = max(traj[:, 1]) + 5
    dpi = 25
    width = (max_x - min_x) / dpi
    # setup plot
    fig = plt.figure(figsize=(width, 6), dpi=dpi)
    ax = fig.add_subplot(111)
    ax.hold(True)
    plt.axis("off")
    ax.set_ylim([min_y, max_y])
    if invert_y:
        plt.gca().invert_yaxis()
    # count penups, we don't want to draw lines between penups and -downs
    stroke_endpoints = np.where(traj[:, 2] == 1)[0]
    begin = 0
    for end in stroke_endpoints:
        plt.plot(traj[begin:end+1, 0], traj[begin:end+1, 1], c='black', lw=15)
        begin = end + 1
    plt.savefig("{}/{}.png".format(outfolder, name), bbox_inches='tight', dpi=dpi)
    plt.close()


def render_trajectory_to_buffer(traj, invert_y=False):
    min_x = min(traj[:, 0]) - 5
    min_y = min(traj[:, 1]) - 5
    max_x = max(traj[:, 0]) + 5
    max_y = max(traj[:, 1]) + 5
    dpi = 25
    width = (max_x - min_x) / dpi
    # setup plot
    fig = plt.figure(figsize=(width, 6), dpi=dpi)
    ax = fig.add_subplot(111)
    ax.hold(True)
    plt.axis("off")
    ax.set_ylim([min_y, max_y])
    if invert_y:
        plt.gca().invert_yaxis()
    # count penups, we don't want to draw lines between penups and -downs
    stroke_endpoints = np.where(traj[:, 2] == 1)[0]
    begin = 0
    for end in stroke_endpoints:
        plt.plot(traj[begin:end+1, 0], traj[begin:end+1, 1], c='black', lw=15)
        begin = end + 1
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # plt.close()
    buf.seek(0)
    im = mimg.imread(buf)
    buf.close()
    return im