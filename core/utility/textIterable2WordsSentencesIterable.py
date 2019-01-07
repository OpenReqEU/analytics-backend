#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import logging
import os
import sys

import nltk
from    nltk.corpus import stopwords
from    nltk.tokenize import RegexpTokenizer

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level=logging.INFO)




def sentence2words(sentence):
    sentence  = sentence.lower() # convert all words into lowercase
    tokens    = RegexpTokenizer(r"(?u)\b\w\w+\b").tokenize(sentence)
    tokens    = [w  for w in tokens if w not in set(stopwords.words('italian'))]
    tokens    = [w  for w in tokens if w not in set(stopwords.words('english'))]
    return tokens



def text2sentences(txt):
    tokenizer = nltk.data.load('tokenizers/punkt/italian.pickle')
    for s in tokenizer.tokenize(txt):
        yield s



def text2words(text):
    words = []
    for sentence in text2sentences(text):
        words += sentence2words(sentence)
    return words



class  TextIterable2WordsSentencesIterable(object):
    def __init__(self, text_iterable):
        self.text_iterable = text_iterable

    def __iter__(self):
        for line in self.text_iterable:
            # split the read line into sentences using NLTK
            for sentence in text2sentences(line):
                # split the sentence into words using regex
                words = sentence2words(sentence)
                #skip short sentences with less than 4 words
                if len(words)<1:
                    continue
                yield words




################################################################################################################################################################
################################################################################################################################################################
################################################################################################################################################################





def tst_sentence2words():
    line= u'fatto farò farai faranno; il tutto e allora migliorerà.  rumore e rumorosità rumoroso rumori.  alfa e beta'
    print sentence2words(line)


def tst_text2sentences():
    line= u'fatto farò farai faranno; il tutto e allora migliorerà.  rumore e rumorosità rumoroso rumori.  alfa e beta'
    for s in text2sentences(line):
        print s



def tst_File2TextIterator():
    file_iterator = File2TextIterable('./data/t.txt')
    for l in file_iterator:
        print l
    for l in file_iterator:
        print l


def tst_TextIterator2WordsSentencesIterator_MemoryConsuming():
    #METODO 1: POCO EFFICIENTE...
    text_iterable = open('./data/t.txt', 'r').readlines()
    for words in TextIterable2WordsSentencesIterable(text_iterable):
        print words



def tst_TextIterator2WordsSentencesIterator_Iterative():
    #METODO 2: OK.... read each line from file (Without reading the entire file)
    text_iterable = File2TextIterable('./data/t.txt')
    for words in TextIterable2WordsSentencesIterable(text_iterable):
        print words









def main():
    # tst_sentence2words()
    # tst_text2sentences()
    # tst_TextIterator2WordsSentencesIterator_MemoryConsuming()
    # tst_TextIterator2WordsSentencesIterator_Iterative()
    tst_File2TextIterator()

    return 0




if __name__ == '__main__':
    main()








