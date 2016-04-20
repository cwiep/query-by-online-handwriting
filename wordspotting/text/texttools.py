# -*- coding: utf-8 -*-
"""
@author: Christian Wieprecht
@license: Apache License, Version 2.0

Helper functions for various text operations.
"""
import math


def get_word_index(trans_data_pages):
    """
    Builds dictionary containing unique words as keys and number of occurences of these words as values.

    @param trans_data_pages: List of pages. Each page is a list containing TransData objects for each word.
    @return: Dictionary of format {'word' => occurences of 'word'}
    """
    words = {}
    for page in trans_data_pages:
        for word in page:
            w = word.word
            words[w] = words.get(w, 0) + 1
    return words


def idf(trans_data_pages, word):
    """
    Calculate inverse document frequency.

    @param trans_data_pages: List of pages. Each page is a list containing TransData objects for each word.
    @param word: String to calculate IDF for.
    @return: Inverse document frequency of word.
    """
    count = 0
    for page in trans_data_pages:
        vocab = get_word_index([page])
        if word in vocab.keys():
            count += 1
    if count == 0:
        return 0.0
    return math.log(len(trans_data_pages) / float(count))


def tf(page, word):
    """
    Calculate term frequency.

    @param page: Page to calculate term frequency for.
    @param word: Word to count frequency of.
    @return: Term frequency of word in page.
    """
    vocab = get_word_index([page])
    return vocab.get(word, 0) / float(len(page))


def tfidf(trans_data_pages, page_index, word):
    """
    Calculate TFIDF value.

    @param trans_data_pages: List of pages. Each page is a list containing TransData objects for each word.
    @param page_index: Index of the page the word is from.
    @param word: Query word.
    @return: TFIDF. Duh.
    """
    return tf(trans_data_pages[page_index], word) * idf(trans_data_pages, word)


def get_n_grams(word, n):
    '''
    Calculates list of ngrams for a given word.

    @param word: Word to calculate ngrams for.
    @param n: Maximal ngram size: n=3 extracts 1-, 2- and 3-grams.
    @return: List of ngrams as strings.
    '''
    return [word[i:i+n]for i in range(len(word)-n+1)]


if __name__ == '__main__':
    pass