#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Christian Wieprecht
@license: Apache License, Version 2.0

Simple demo interface to test the presented method.
"""
import numpy as np
import cv2

from demo_interface import QboWordspotting


inside_continue = lambda x, y: 1120 < x < 1270 and 490 < y < 590
inside_remove = lambda x, y: 960 < x < 1110 and 490 < y < 590


class WriteSurface:
    def __init__(self):
        self.image = np.zeros((600, 1280, 3), np.uint8)
        self.last_point = (0, 0)
        self.pendown = False
        self.points = []
        self.threshold = 10
        self.wordspotting = QboWordspotting()
        self.reset()

    def reset(self):
        self.image = np.zeros((600, 1280, 3), np.uint8)
        self.points = []
        # query retrieval start button
        cv2.rectangle(self.image, (1120, 490), (1270, 590), color=(0, 255, 0), thickness=-1)
        # clear query button
        cv2.rectangle(self.image, (960, 490), (1110, 590), color=(0, 0, 255), thickness=-1)
        self.last_point = (0, 0)

    def draw_stroke(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if inside_continue(x, y):
                self.wordspotting.retrieve_hits(self.points)
                return
            if inside_remove(x, y):
                self.reset()
                return
            self.pendown = True
            self.points.append([x, y, 0])
            self.last_point = (x, y)
        if event == cv2.EVENT_MOUSEMOVE:
            if self.pendown:
                if abs(self.last_point[0] - x) > self.threshold or abs(self.last_point[1] - y) > self.threshold:
                    # cv2.circle(self.image,(x,y), 5,(255, 0, 0),-1)
                    cv2.line(self.image, self.last_point, (x, y), (255, 255, 255), thickness=6)
                    self.points.append([x, y, 0])
                    self.last_point = (x, y)
            return
        if event == cv2.EVENT_LBUTTONUP:
            if not inside_continue(x, y):
                self.pendown = False
                self.points.append([x, y, 1])
                self.last_point = (x, y)


# Create a black image, a window and bind the function to window
write_surface = WriteSurface()
cv2.namedWindow('QbO Wordspotting (Escape to quit)')
cv2.setMouseCallback('QbO Wordspotting (Escape to quit)', write_surface.draw_stroke)

while 1:
    cv2.imshow('QbO Wordspotting (Escape to quit)', write_surface.image)
    if cv2.waitKey(20) & 0xFF == 27:
        break
cv2.destroyAllWindows()