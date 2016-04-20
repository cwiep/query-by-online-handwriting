# -*- coding: utf-8 -*-
"""
@author: Christian Wieprecht
@license: Apache License, Version 2.0

Loading transcription data used in trans(cription) data pages throughout this project.
"""

class WordData:
    def __init__(self, line, page_basename=None):
        # line format is "xstart ystart xend yend word"
        parts = line.split(" ")
        self.__setup(int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]), parts[4].strip())
        self.page_basename = page_basename

    def __setup(self, xstart, ystart, xend, yend, word):
        self.xstart = xstart
        self.ystart = ystart
        self.xend = xend
        self.yend = yend
        self.word = word
        self.width = xend - xstart
        self.height = yend - ystart

    def __str__(self):
        return "{}, {}, '{} {} {} {}'".format(self.word, self.page_basename, self.xstart, self.ystart, self.xend, self.yend)

    def point_in_rect(self, px, py):
        return self.xstart < px < self.xend and self.ystart < py < self.yend


def _load_transcription_data_file(filename, page_basename=None):
    """
    Create WordData object for each word in page.

    @param filename: Path to page.
    @return: List of WordData objects.
    """
    trans_data = []
    with open(filename, "r") as trans_file:
        for line in trans_file:
            t = WordData(line, page_basename)
            trans_data.append(t)
    return trans_data


def load_transcription_data(folder, filenames):
    """
    Create WordData list for all files in 'filenames'. Filetype extension is '.gtp' and has to be omitted!
    @param folder: Folder with page files.
    @param filenames: List of page files withput filetype extension.
    @return: List of pages, where each page is a list of WordData objects.
    """
    return [_load_transcription_data_file("{}/{}.gtp".format(folder, t)) for t in filenames]


def create_fake_transcription_data(filenames):
    """
    Create WordData list for all filenames. The annotation is read from the filename
    with format id_annotation.txt.
    """
    return [[WordData("0 0 0 0 {}".format(word))] for word in [f.split("_")[1] for f in filenames]]