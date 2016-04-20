# -*- coding: utf-8 -*-
"""
@author: Christian Wieprecht
@license: Apache License, Version 2.0

Online handwriting normalization methods.
"""
import numpy as np
from scipy.stats import linregress
from scipy.special import binom
import math

import tools.logging as log


def move_to_origin(traj):
    """
    Move trajectory so that the lower left corner
    of its bounding box is the origin afterwards.
    """
    min_x = min(traj[:, 0])
    min_y = min(traj[:, 1])
    return traj - [min_x, min_y, 0]


def flip_vertically(traj):
    """
    Rotates trajectory by 180 degrees.
    """
    max_y = max(traj[:, 1])
    return np.array([[x, max_y - y, p] for [x, y, p] in traj])


def correct_slope(traj):
    """
    Rotates trajectory so that the regression line through
    all points is the horizontal line afterwards.
    """
    [slope, intercept, _, _, _] = linregress(traj[:, :2])
    alpha = math.atan(-slope)
    cos_alpha = math.cos(alpha)
    sin_alpha = math.sin(alpha)
    min_x = min(traj[:, 0])
    min_y = min(traj[:, 1])
    rot_x = lambda x, y: min_x + cos_alpha * (x - min_x) - sin_alpha * (y - min_y)
    rot_y = lambda x, y: min_y + sin_alpha * (x - min_x) + cos_alpha * (y - min_y)
    new_traj = np.array([[rot_x(x, y), rot_y(x, y), p] for [x, y, p] in traj])
    new_min_x = min(new_traj[:, 0])
    new_min_y = min(new_traj[:, 1])
    return new_traj - [new_min_x, new_min_y, 0]


def correct_slant(traj):
    """
    Removes the most dominant slant-angle from the trajectory.
    """
    last_point = traj[0]
    angles = []
    for cur_point in traj[1:]:
        if last_point[2] == 1:
            # don't measure angles for "invisible" lines
            last_point = cur_point
            continue
        if (cur_point[0] - last_point[0]) == 0:
            angles.append(90)
        else:
            angle = math.atan((cur_point[1] - last_point[1]) / float(cur_point[0] - last_point[0])) * 180 / math.pi
            angles.append(int(angle))
        last_point = cur_point
    # print("found {} angles for {} points".format(len(angles), len(traj)))
    angles = np.array(angles) + 90
    bins = np.bincount(angles, minlength=181)
    # weighting all angles with discrete standard gaussian distribution
    weights = [binom(181, k)/181.0 for k in range (1, 182)]
    weights /= sum(weights)
    bins = bins.astype(float) * weights
    # smoothing entries with neighbours, first and last points remain unchanged
    gauss = lambda p, c, n: 0.25 * p + 0.5 * c + 0.25 * n
    smoothed = [bins[0]] + [gauss(bins[i-1], bins[i], bins[i+1]) for i in range(len(bins)-1)] + [bins[len(bins)-1]]
    slant = np.argmax(smoothed) - 90
    # print("slant is {}".format(slant))
    # print(len(smoothed))
    min_x = min(traj[:, 0])
    min_y = min(traj[:, 1])
    rotate = lambda x, y: min_x + (x - min_x) - math.tan(slant * math.pi / 180) * (y - min_y)
    return np.array([[rotate(x, y), y, p] for [x, y, p] in traj])


def resampling(traj, step_size=5):
    """
    Replaces given trajectory by a recalculated sequence of equidistant points.
    """
    t = []
    t.append(traj[0, :])
    i = 0
    length = 0
    current_length = 0
    old_length = 0
    curr, last = 0, None
    len_traj = traj.shape[0]
    while i < len_traj:
        current_length += step_size
        while length <= current_length and i < len_traj:
            i += 1
            if i < len_traj:
                last = curr
                curr = i
                old_length = length
                length += math.sqrt((traj[curr, 0] - traj[last, 0])**2) + math.sqrt((traj[curr, 1] - traj[last, 1])**2)
        if i < len_traj:
            c = (current_length - old_length) / float(length-old_length)
            x = traj[last, 0] + (traj[curr, 0] - traj[last, 0]) * c
            y = traj[last, 1] + (traj[curr, 1] - traj[last, 1]) * c
            p = traj[last, 2]
            t.append([x, y, p])
    t.append(traj[-1, :])
    return np.array(t)


def normalize_height(traj, new_height=150):
    """
    Returns scaled trajectory whose height will be new_height.
    TODO: try to scale core height instead
    """
    min_y = min(traj[:, 1])
    max_y = max(traj[:, 1])
    old_height = max_y - min_y
    scale_factor = new_height / float(old_height)
    traj[:, :2] *= scale_factor
    return traj


def smoothing(traj):
    """
    Applies gaussian smoothing to the trajectory with a (0.25, 0.5, 0.25) sliding
    window. Smoothing point p(t) uses un-smoothed points p(t-1) and p(t+1).
    """
    s = lambda p, c, n: 0.25 * p + 0.5 * c + 0.25 * n
    smoothed = np.array([s(traj[i-1], traj[i], traj[i+1]) for i in range(1, traj.shape[0]-1)])
    # the code above also changes penups, so we just copy them again
    smoothed[:, 2] = traj[1:-1, 2]
    # we deleted the unsmoothed first and last points,
    # so the last penup needs to be moved to the second to last point
    smoothed[-1, 2] = 1
    return smoothed


def remove_delayed_strokes(traj):
    """
    Removes points of delayed strokes (segments between two penups)
    from the trajectory. Removal if right edge of stroke's bounding box
    is to the left of the right edge of the last non-delayed stroke.
    """
    stroke_endpoints = np.where(traj[:, 2] == 1)[0]
    # first stroke is by convention never delayed
    begin = stroke_endpoints[0] + 1
    new_traj = []
    new_traj.extend(traj[:begin, :])
    delayed = []
    # delayed strokes must begin and end left of the current orientation point
    orientation_point = traj[begin-1, :2]
    for end in stroke_endpoints[1:]:
        stroke = traj[begin:end+1, :]
        max_x = max(stroke[:, 0])
        begin = end + 1
        if max_x >= orientation_point[0]:
            new_traj.extend(stroke)
            orientation_point = traj[begin-1, :2]
        else:
            delayed.append(stroke)
    return np.array(new_traj), np.array(delayed)


def normalize_trajectory(traj, args):
    """
    Applies given normalization steps in args to trajectory of points in traj.
    Valid normalizations are "flip", "slope", "origin", "resample", "slant", "height",
    "smooth" and "delayed". Note that with application of "delayed" there will be
    two objects returned, the trajectory and the list of delayed strokes.

    The object that "traj" points to WILL BE CHANGED!
    """
    if "flip" in args:
        traj = flip_vertically(traj)
    if "slope" in args:
        traj = correct_slope(traj)
    if "origin" in args:
        traj = move_to_origin(traj)
    if "resample" in args:
        traj = resampling(traj)
    if "slant" in args:
        traj = correct_slant(traj)
    if "height" in args:
        traj = normalize_height(traj)
    if "smooth" in args:
        traj = smoothing(traj)
    if "delayed" in args:
        traj, delayed = remove_delayed_strokes(traj)
        return traj, delayed
    return traj
