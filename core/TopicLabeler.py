#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 11:21:03 2018

@author: francesco pareo
@location: bologna
"""

# import
from __future__ import division

from gensim.models import Word2Vec
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import json
from collections import Counter

from clustering_library import scatter_plot_nD, optimalK
from word_freq_library import idf

def textRanking(keywords, model_w2v):
    vec_size = model_w2v.vector_size

    # extract mean w2v vector
    mean_w2v = np.zeros((len(keywords), vec_size), dtype='float32')
    for _n, _doc in enumerate(keywords):
        _sum_w2v = np.zeros(vec_size, dtype='float32')
        _n_key = 0
        for _n_key, _key in enumerate(_doc):
            try:
                _sum_w2v = model_w2v.wv[_key] + _sum_w2v
                _n_key += 1
            except KeyError:
                pass
        mean_w2v[_n, :] = _sum_w2v / _n_key

    mean_w2v[np.isnan(mean_w2v).any(axis=1), :] = np.zeros(vec_size, dtype='float32')

    k_cluster_range = range(2, 15)
    best_k, res_dict = optimalK(k_cluster_range, mean_w2v, verbose=True, getPlot=True)

    print('best k:', best_k)
    best_k_r = int(round(best_k, 0))
    print('cluster distribution, for k= ' + str(best_k_r))
    print(Counter(res_dict['labels'][best_k_r]))


    # %%  find best cluster
    km = KMeans(init='k-means++', n_clusters=best_k_r, n_init=10).fit(mean_w2v)

    # %% FIND TOPICS
    nClosest = 20  # num of words
    topic_dict = {}  # d=defaultdict(dict)
    key_list = list()

    for n, k in enumerate(k_cluster_range):
        labels = res_dict['labels'][n]
        label_set = list(set(labels))
        print('#### k:', k)
        centroid_word = list()
        l = [None] * (len(k_cluster_range) + 1)
        for n2, label in enumerate(label_set):
            closestWord = idf([elem for n, elem in enumerate(keywords) if labels[n] == label])
            only_words = [word[0] for word in closestWord[0:nClosest]]
            centroid_word.append(only_words)
            l[n2] = only_words
        key_list.append(l)
        topic_dict[str(k)] = centroid_word

    return topic_dict[str(int(best_k))]


