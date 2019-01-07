import pandas as pd
import logging
import os
import numpy

#customs
import core.utility.logger as logger
from core.TextCleaner import TextCleaner
import core.configurations
import re
from core.utility.serializator import save_obj, load_obj
from core.utility import utilities

conf = core.configurations.get_conf()
log_path = conf.get('MAIN', 'log_path')
log_file_name = conf.get('MAIN', 'log_file_name')
log = logger.getLogger(__name__, log_path , log_file_name)

# create a list of cleaned text
def cleanText(input_list):
    # declare and init textCleaner
    text_cleaner = TextCleaner(rm_punct = True, rm_digits = True, rm_hashtags = False)

    # cleaning text
    cleaned_list = []
    for text in input_list:
        if (str(text) != "nan"):
            text = text.replace(" /", "/")
            text = re.sub(r'\b\w{1,3}\b', '', text)
            cleaned_list.append(text_cleaner.regex_applier(text))

    #REMOVE STOPWORDS
    no_stopwords_sentences = []
    stopwords_dict = stopwordsDictFromFile(conf.ConfigSectionMap('STOPWORDS_FILES'))
    for text in cleaned_list:
        if (str(text) != "nan"):
            words_no_stopwords = []
            words = utilities.convertSentenceToListOfWords(text)
            for word in words:
                if not stopwords_dict.has_key(word.lower().encode("utf-8")):
                    words_no_stopwords.append(word)
            no_stopwords_sentences.append(utilities.concatWords(words_no_stopwords))


    return no_stopwords_sentences


def stopwordsDictFromFile(list_files_path):

    fun_lower = lambda x: x.lower()
    vec_fun_lower = numpy.vectorize(fun_lower)
    stopwords_dict = {}
    for file_path in list_files_path:
        array_stopwords = numpy.array(open(file_path).read().splitlines())
        array_stopwords = vec_fun_lower(array_stopwords)
        stopwords_dict.update(dict(zip(array_stopwords.tolist(), (numpy.zeros(shape=array_stopwords.shape)*numpy.nan).tolist())))

    return stopwords_dict