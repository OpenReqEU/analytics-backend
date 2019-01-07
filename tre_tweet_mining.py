#!/usr/bin/env python2
"""
Tweet Text Mining
Claudia Rapuano - Roma
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
import sys
import os
import numpy as np

# custom functions
import core.Annotation_Library
import core.WikipediaExtractorLibrary
from core.utility.serializator import save_obj, load_obj
from core.micro_services.word2vec_ms import  *
import core.utility.logger as logger
import core.configurations
from core.micro_services import clean_text_ms, som_ms, word2vec_ms
import core.corpus as Corpus

conf = core.configurations.get_conf()
log_path = conf.get('MAIN', 'log_path')
log_file_name = conf.get('MAIN', 'log_file_name')
log = logger.getLogger(__name__, log_path , log_file_name)


USE_WIKIPEDIA_FOR_W2V = conf.get('FLAG', 'USE_WIKIPEDIA_FOR_W2V')

def main():
    log.info("----------------------------------START------------------------------------")
    reload(sys)
    sys.setdefaultencoding('utf-8')

    document_path_file = conf.get('MAIN', 'path_document')
    log.info("reading input file: "+document_path_file)

    # ------------------------READ INPUT-------------------------------------------------------
    # read csv into list of string
    input_list = pd.read_csv(document_path_file, encoding='utf-8', error_bad_lines=False)
    # read which rows are from twitter
    source_value = np.array(input_list['idriferimento_ricerca'])
    tweet_rows_bool = (source_value == 5) | (source_value == 6)
    # read all input
    input_list = input_list['messaggio'].tolist()

    # ------------------------CLEANING TEXT---------------------------------------------------
    cleaned_input_list = []
    # read csv file
    path_csv_output_folder = conf.get('MAIN', 'path_csv_output_folder')
    file = path_csv_output_folder + 'cleaned_tweet_list.csv'
    if (os.path.isfile(file)):
        log.info("reading input from file " + file)
        cleaned_input_list = pd.read_csv(file, encoding='utf-8', error_bad_lines=False)
        cleaned_input_list = cleaned_input_list['colummn'].tolist()

    if (cleaned_input_list == [] or cleaned_input_list == [[]]):
        log.info("CLEANING TEXT")
        cleaned_input_list = clean_text_ms.cleanText(input_list)

        # write output to csv
        df = pd.DataFrame(cleaned_input_list, columns=["colummn"])
        df.to_csv(file, index=False)
        log.info("file saved in " + file)

    # if word2vec does not exist or rebuild is setted train w2v model
    if not os.path.exists(conf.get('MAIN', 'path_pickle_w2v_model')):
        #-------------------------GET ENTITIES----------------------------------------------------
        log.info("GET ENTITIES")

        entity_list = []
        file_entity_list = path_csv_output_folder+'entity_list.csv'
        file = file_entity_list
        if (os.path.isfile(file)):
            log.info("reading input from file " + file)
            # read csv file
            entity_list = pd.read_csv(file, encoding='utf-8', error_bad_lines=False)
            entity_list = entity_list['colummn'].tolist()

        tweet_with_entity_list = []
        file_tweet_with_entity_list = path_csv_output_folder+'tweet_with_entity_list.csv'
        file = file_tweet_with_entity_list
        if (os.path.isfile(file)):
            log.info("reading input from file " + file)
            tweet_with_entity_list = pd.read_csv(file, encoding='utf-8', error_bad_lines=False)
            tweet_with_entity_list = tweet_with_entity_list['colummn'].tolist()

        all_uri = []
        file_all_uri = path_csv_output_folder+'all_uri.csv'
        file = file_all_uri
        if (os.path.isfile(file)):
            log.info("reading input from file " + file)
            all_uri = pd.read_csv(file, encoding='utf-8', error_bad_lines=False)
            all_uri = all_uri['colummn'].tolist()

        # get entities
        if (entity_list == [] or entity_list == [[]]):
            confidence = conf.get('ENTITY', 'confidence')
            entity_list, tweet_with_entity_list, all_uri = Corpus.getEntities(cleaned_input_list, confidence=confidence)

            file = file_entity_list
            # write output to csv
            df = pd.DataFrame(entity_list, columns=["colummn"])
            df.to_csv(file, index=False)
            log.info("file saved in " + file)

            file = file_tweet_with_entity_list
            df = pd.DataFrame(tweet_with_entity_list, columns=["colummn"])
            df.to_csv(file, index=False)
            log.info("file saved in " + file)

            file = file_all_uri
            df = pd.DataFrame(all_uri, columns=["colummn"])
            df.to_csv(file, index=False)
            log.info("file saved in " + file)

        with_wiki_pages = conf.get('MAIN', 'with_wiki_pages')
        if (with_wiki_pages == 'False'):
            corpus = cleaned_input_list
        else:
            #-------------------------GET WIKIPEDIA PAGES---------------------------------------------
            log.info("GET WIKIPEDIA PAGES")
            wikipage_list = []

            wikipage_list_file = path_csv_output_folder + 'wikipage_list.csv'
            file = wikipage_list_file
            if (os.path.isfile(file)):
                log.info("reading input from file " + file)
                # read csv file
                wikipage_list = pd.read_csv(file, encoding='utf-8', error_bad_lines=False)
                wikipage_list = wikipage_list['colummn'].tolist()

            # get wikipedia page
            if (wikipage_list == [] or wikipage_list == [[]]):
                wikipage_list = Corpus.getWikipediaPages(all_uri)
                wikipage_list = clean_text_ms.cleanText(wikipage_list)

                # write csv
                df = pd.DataFrame(wikipage_list, columns=["colummn"])
                df.to_csv(file, index=False)
                log.info("file saved in " + file)

            #-------------------------CREATE CORPUS---------------------------------------------------
            print log.info("CREATE CORPUS")
            corpus = []

            # read csv file
            corpus_file = path_csv_output_folder + 'corpus.csv'
            file = corpus_file
            if (os.path.isfile(file)):
                log.info("reading input from file " + file)
                # read csv file
                corpus = pd.read_csv(file, encoding='utf-8', error_bad_lines=False)
                corpus = corpus['colummn'].tolist()

            # create corpus
            if (corpus == [] or wikipage_list == [[]]):
                tweet_corpus = Corpus.createTweetCorpus(wikipage_list, cleaned_input_list, tweet_with_entity_list)
                corpus = tweet_corpus
                if (USE_WIKIPEDIA_FOR_W2V):
                    corpus += wikipage_list

        corpus_file = path_csv_output_folder + 'corpus.csv'
        file = corpus_file
        # write corpus to csv
        df = pd.DataFrame(corpus, columns=["colummn"])
        df.to_csv(file, index=False)
        log.info("file saved in " + file)

        #-------------------------TRAIN MODEL W2V-------------------------------------------------
        # train model W2v
        log.info("TRAIN W2V")
        trainW2Vmodel(corpus)

    #----------------------TRAINING SOM------------------------------------------------
    # load trained model W2V
    w2v_model = Word2Vec.load(conf.get('MAIN', 'path_pickle_w2v_model'))
    log.info("loading W2V model "+conf.get('MAIN', 'path_pickle_w2v_model'))

    # get w2v words, dict words and vectors only for tweet
    embedded_words_t_w, dict_index2word_t_w, dict_word2indext_w = collectWords(w2v_model)

    # train SOM: get codebook matrix
    doTrainSom = conf.getboolean('ADVANCED_ASOM', 'do_trainSom')
    if doTrainSom or not os.path.exists(conf.get('MAIN', 'path_pickle_som_model')):
        width = int(conf.get('ADVANCED_ASOM', 'width'))
        height = int(conf.get('ADVANCED_ASOM', 'height'))
        empty_codebook_threshold = int(conf.getboolean('ADVANCED_ASOM', 'empty_codebook_threshold'))

        log.info("training som [" + str(width) + "x" + str(height) + "]")
        mySom = som_ms.trainSOM(embedded_words_t_w, dict_index2word_t_w, conf, width, height)

        min_size_codebook_mtx = int(conf.get('ADVANCED_ASOM', 'min_size_codebook_mtx'))
        step_codebook_mtx = int(conf.get('ADVANCED_ASOM', 'step_codebook_mtx'))

        # decrease som dimensions if we have more than one codebook empty
        while (not som_ms.isGoodResult(mySom, width, height, empty_codebook_threshold) and width > min_size_codebook_mtx+step_codebook_mtx):
            log.info("training som ["+str(width)+"x"+str(height)+"]")
            width = height = height-2
            mySom = som_ms.trainSOM(embedded_words_t_w, dict_index2word_t_w, conf, width, height)

        save_obj(mySom, conf.get('MAIN', 'path_pickle_som_model'))

    #--------- PREDICT: only on tweets------------------------------------------------

    cleaned_input_list = clean_text_ms.cleanText(input_list)

    # getting only tweets of 3
    cleaned_tweet_rows = []
    tweet_rows = []
    index = 0
    for item in tweet_rows_bool:
        if item == True:
            cleaned_tweet_rows.append(cleaned_input_list[index])
            tweet_rows.append(input_list[index])
        index = index+1

    # get embedded words from input
    embedded_words, dict_index2word, dict_word2index = word2vec_ms.getEmbeddedWords(
        cleaned_tweet_rows)

    word2VecMS = Word2VecMS(tweet_rows, w2v_model)
    word2VecMS.computeWord2Tweets()
    word2VecMS.saveObject()

    # load SOM
    mySom = load_obj(conf.get('MAIN', 'path_pickle_som_model'))
    log.info("loading SOM model " + conf.get('MAIN', 'path_pickle_som_model'))

    # predict SOM codebooks and plot
    file_name = conf.get('MAIN', 'MST_html_output_file')
    url = som_ms.doSomAndPlot(mySom, embedded_words, dict_index2word, file_name, "netx")
    file_name = conf.get('MAIN', 'MST_html_d3_output_file')
    url = som_ms.doSomAndPlot(mySom, embedded_words, dict_index2word, file_name, "d3")

    #--------------------PLOT/PRINT INFO ON SOM---------------------------------
    png = som_ms.getCodebookActivation()

    num_of_topic = conf.getint('GRAPH_IMG', 'num_of_topic_for_frequencies')
    html = som_ms.getCellFrequencyDistribution(cleaned_tweet_rows, w2v_model, mySom, num_of_topic, 'bar')
    html = som_ms.getCellFrequencyDistribution(cleaned_tweet_rows, w2v_model, mySom, num_of_topic, 'bubble')

    png = som_ms.getUmatrix()
    plt.show()

    print som_ms.getCostOfSom()

    #-------------------------KMEANS --------------------------------------------------
    if not os.path.exists(conf.get('MAIN', 'path_pickle_cluster_model')):
        log.info("START CLUSTERING")
        mySom = load_obj(conf.get('MAIN', 'path_pickle_som_model'))
        make_figure = False
        mySom.fit_cluster(cluster_model         =   None,
                          num_cluster_min       =   conf.getint('ADVANCED_ASOM', 'num_cluster_min'),
                          num_cluster_max       =   conf.getint('ADVANCED_ASOM', 'num_cluster_max'))

        save_obj(mySom.cluster_model, conf.get('MAIN', 'path_pickle_cluster_model'))
        log.info("saved cluster model in "+conf.get('MAIN', 'path_pickle_cluster_model'))

    # make clustering and plot
    file_name = conf.get('MAIN', 'MST_cluster_csv_output_file')
    url = som_ms.doClusteringAndPlot(cleaned_tweet_rows, file_name)

def convertToListOfWords(list_unicode):
    # convert unicode to string
    #list_of_string = list(map(lambda (tweet): tweet.encode('UTF8'), list_unicode))
    list_of_string = []
    for tweet in list_unicode:
        list_of_string.append(str(tweet).encode('UTF8'))
    # get all words of tweets
    joined_sentences = ' '.join(list_of_string)
    list_of_words = joined_sentences.split(' ')
    return list_of_words

# given a list of list count the number of empty list
def countEmptyList(data):
    count = 0
    for i in data:
        if (i == []):
            count += 1
    return count



if __name__ == '__main__':
    main()
