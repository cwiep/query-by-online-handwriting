# -*- coding: utf-8 -*-
"""
@author: Christian Wieprecht
@license: Apache License, Version 2.0
"""
import numpy as np
from matplotlib.patches import Circle, Rectangle
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import linregress


def visualize_descriptors(image, keyp, labels, n_centroids, cell_size=5, draw_cells=False, draw_spatial_pyramid=False):
    # setup plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(image, cmap=cm.get_cmap('Greys_r'))
    ax.hold(True)
    ax.autoscale(enable=False)
    ax.axis("off")
    # draw colored points
    colormap = cm.get_cmap('jet')
    desc_len = cell_size * 4
    for (x, y), label in zip(keyp, labels):
        color = colormap(label / float(n_centroids))
        # ax.text(x, y, str(label))
        if draw_cells:
            rect = Rectangle((x - desc_len / 2, y - desc_len / 2), desc_len, desc_len, alpha=0.6, lw=1, color="#7e94b9")
            for p_factor in [0.25, 0.5, 0.75]:
                offset_dyn = desc_len * (0.5 - p_factor)
                offset_stat = desc_len * 0.5
                line_h = Line2D((x - offset_stat, x + offset_stat), (y - offset_dyn, y - offset_dyn), alpha=0.3, lw=1)
                line_v = Line2D((x - offset_dyn, x - offset_dyn), (y - offset_stat, y + offset_stat), alpha=0.3, lw=1)
                ax.add_line(line_h)
                ax.add_line(line_v)
            ax.add_patch(rect)
        circle = Circle((x, y), radius=1, fc='#d52d31', ec='#d52d31', alpha=1)
        ax.add_patch(circle)
        if draw_spatial_pyramid:
            im_width = image.shape[1]
            im_height = image.shape[0]
            w = int(im_width / 3)
            line_vert = lambda x: Line2D((w*x, w*x), (0, im_height), alpha=0.08, lw=2, color='red')
            h = int(im_height / 2)
            line_hor = lambda y: Line2D((0, im_width), (h*y, h*y), alpha=0.08, lw=2, color='red')
            ax.add_line(line_vert(1))
            ax.add_line(line_vert(2))
            ax.add_line(line_hor(1))
    plt.show()


def draw_trajectories(traj_list, draw_reg=False, invert_y=False):
    min_x = min(traj_list[0][:, 0]) - 20
    max_x = max(traj_list[0][:, 0]) + 20
    dpi = 50
    width = (max_x - min_x) / dpi
    fig = plt.figure(figsize=(width, 6), dpi=dpi)
    ax = fig.add_subplot(111)
    ax.hold(True)
    plt.axis("equal")
    plt.axis("off")
    # plt.axis('off')
    if invert_y:
        plt.gca().invert_yaxis()
    colormap = cm.get_cmap('jet')
    for idx, traj in enumerate(traj_list):
        [slope, intercept, _, _, _] = linregress(traj[:, :2])
        color = colormap(idx / float(len(traj_list)))
        stroke_endpoints = np.where(traj[:, 2] == 1)[0]
        begin = 0
        col = 'black' if idx%2 == 0 else 'blue'
        for i, end in enumerate(stroke_endpoints):
            plt.plot(traj[:, 0], traj[:, 1], c="black", lw=5, marker=".")
            begin = end + 1
        if draw_reg:
            f = lambda x: slope * x + intercept
            minx, maxx = min(traj[:, 0]), max(traj[:, 0])
            plt.plot([minx, maxx], [f(minx), f(maxx)], c='r', lw=1)
            ax.annotate(str(idx), xy=(maxx, f(maxx)), size=14)
    plt.show()


def draw_segments(segments, traj, invert_y=False):
    min_x = min(traj[:, 0]) - 20
    max_x = max(traj[:, 0]) + 20
    dpi = 50
    width = (max_x - min_x) / dpi
    fig = plt.figure(figsize=(width, 6), dpi=dpi)
    ax = fig.add_subplot(111)
    ax.hold(True)
    plt.axis("equal")
    # plt.axis('off')
    if invert_y:
        plt.gca().invert_yaxis()
    colormap = cm.get_cmap('jet')
    for idx, segment in enumerate(segments):
        # stroke_endpoints = np.where(traj[:, 2] == 1)[0]
        # begin = 0
        # for i, end in enumerate(stroke_endpoints):
        plt.plot(segment[:, 0], segment[:, 1], c=colormap(idx / float(len(segment))), lw=15)
        # begin = end + 1
    plt.show()


def plot_query_results(images, top_words, word_num_offsets, trans_data_pages, query_word, save_to_pdf=False):
    num_pages = len(images)
    # setup plot
    for p in range(num_pages):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(images[p], cmap=cm.get_cmap('Greys_r'))
        ax.hold(True)
        ax.axis('off')
        ax.autoscale(enable=False)
        for rank, (i, _, _) in enumerate(top_words):
            # get label- and keypoint index
            if not (word_num_offsets[p] <= i < word_num_offsets[p+1]):
                continue
            word_data = trans_data_pages[p][i-word_num_offsets[p]]
            w = word_data.xend - word_data.xstart
            h = word_data.yend - word_data.ystart
            col = 'green' if word_data.word == query_word else 'red'
            rect = Rectangle((word_data.xstart, word_data.ystart), w, h, alpha=0.2, lw=2, color=col)
            ax.annotate(str(rank), xy=(word_data.xstart, word_data.ystart), size=8)
            ax.add_patch(rect)
        if save_to_pdf:
            plt.savefig('result_{}.pdf'.format(p), bbox_inches='tight', dpi=200)
    if not save_to_pdf:
        plt.show()