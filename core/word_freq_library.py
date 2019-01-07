#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 11:18:53 2018

@author: fra
"""

import operator

def idf(doc_list):
    word_dict = {}
    for doc in doc_list:
        doc = list(set(doc))
        for word in doc:
            word_dict[word] = word_dict.get(word,0)+1
            
    word_dict = sorted(word_dict.items(), key=operator.itemgetter(1),  reverse=True)
    return word_dict


def absf(doc_list):
    word_dict = {}
    for doc in doc_list:
        for word in doc:
            word_dict[word] = word_dict.get(word,0)+1
    return word_dict


def tf(doc_list):
    word_dict = {}
    for doc in doc_list:
        for word in doc:
            word_dict[word] = word_dict.get(word,0)+1
    return word_dict


def epure_dict(word_dict, fa):
    return {k:v for k, v in word_dict.iteritems() if v>fa}
