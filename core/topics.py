from gensim.models.word2vec import Word2Vec
from collections import OrderedDict
from core.AdvancedSOM import ASOM
import os, math
from core.utility.textIsANum import textIsANum
import numpy as np
try:
    import pylab
except:
    pass

# custom functions
from core.utility.serializator import save_obj, load_obj
import core.utility.logger as logger
import core.configurations
from core.micro_services import som_ms, word2vec_ms, clean_text_ms

import matplotlib.pyplot as plt


conf = core.configurations.get_conf()
log_path = conf.get('MAIN', 'log_path')
log_file_name = conf.get('MAIN', 'log_file_name')
log = logger.getLogger(__name__, log_path , log_file_name)

def getNArgMin(n, array):

    min_index = []
    if array.shape[0] < n:
        return -1
    for i in xrange(array.shape[0]):
        min_index.append(array.argmin())
        array[min_index[i]] += np.inf
        if (i + 1)  == n:
            break
    return min_index

def getMultipleValuesFromArray( indexes_values, array):

    values = []
    for index in indexes_values:
        values.append(array[index])
    return values

def getTopicMatrix(topic_index, topics, data2cluster, X, dict_words, conf):

    words_list = []
    matrix_list = []
    for word in getReducedTopics(topic_index, data2cluster, topics, dict_words, conf):
        if dict_words.has_key(word) and len(word) > 2:
            words_list.append(word)
            matrix_list.append(X[dict_words[word],:])

    return np.array(matrix_list), words_list

def stopwordsDictFromFile(list_files_path):

    fun_lower = lambda x: x.lower()
    vec_fun_lower = np.vectorize(fun_lower)
    stopwords_dict = {}
    for file_path in list_files_path:
        array_stopwords = np.array(open(file_path).read().splitlines())
        array_stopwords = vec_fun_lower(array_stopwords)
        stopwords_dict.update(dict(zip(array_stopwords.tolist(), (np.zeros(shape=array_stopwords.shape)*np.nan).tolist())))

    return stopwords_dict

def getReducedTopics(cluster_num, data2cluster, topics, dict_words, conf):

    stopwords_dict = {}
    #stopwords_dict = load_obj(conf.get('MAIN', 'path_pickle_stopwords_dict'))

    bestTopics = []
    offTopics  = []
    for topic_index in list(*np.where(data2cluster.flatten() == cluster_num)):
        for word in topics[topic_index]:
            word = word.replace('.|;|,|:|?|!|(|)','')
            if dict_words.has_key(word) and len(word) > 1 and not stopwords_dict.has_key(word.lower()):
                 bestTopics.append(word.lower())
            else:
                offTopics.append(word.lower())
    return bestTopics

def dryTopics(topics, data2cluster, X, dict_words, dict_X, min_cluster, conf):

    num_epoch = 200
    dried_topics = OrderedDict({})
    for topic_index in xrange(data2cluster.max()+1):

        M, words_list = getTopicMatrix(topic_index, topics, data2cluster, X, dict_words, conf)

        print 'Matrix shape ' + str(M.shape)
        print 'Num of words ' + str(len(words_list))

        if len(words_list) > 20:

            # num_units = int( math.ceil(math.sqrt(M.shape[0])/10.0))
            num_units = int( math.ceil(math.sqrt(M.shape[0])/8))
            print num_units
            somTopic = ASOM(num_units                   = num_units,
                                   outlier_unit_threshold   = conf.getfloat('ADVANCED_ASOM', 'outlier_unit_threshold'),
                                   outlier_percentile       = conf.getfloat('ADVANCED_ASOM', 'outlier_percentile'),
                                   Koutlier_percentile      = conf.getint('ADVANCED_ASOM', 'Koutlier_percentile'),
                                   learning_rate_percentile = conf.getfloat('ADVANCED_ASOM', 'learning_rate_percentile'),
                                   distance_metric          = 'euclidean',
                                    initialize_unit_with_cluster=None
                                   )

            somTopic.train_batch(M, num_epoch=num_epoch)

            dried_topics[topic_index] = []
            list_min_index = []
            for som_index in xrange(somTopic.num_units):
                min_indexes = getNArgMin(conf.getint('TOPICS', 'num_nearest_words'), somTopic.data2dist[somTopic.data2unit == som_index])
                if min_indexes == -1:
                    continue
                min_indexes = getMultipleValuesFromArray(min_indexes, np.where(somTopic.data2unit == som_index)[0])
                dried_topics[topic_index].append([words_list[min_index] for min_index in min_indexes])
        else:
            dried_topics[topic_index] = [getReducedTopics(topic_index, data2cluster, topics, dict_words, conf)]

    return dried_topics

def getTopics(trained_som, X, dict_X):
    data2unit, data2cell, data2dist, data2saliency, data2saliency_index, data2maps = trained_som.predict(X)

    dict_topic = OrderedDict({})
    for cluster_k in xrange(trained_som.num_units):
        index_similar_words = list(np.where((data2unit == cluster_k))[0])
        dict_topic[cluster_k] = []
        print 'topic_%d'% cluster_k
        for index in index_similar_words:
            if len(dict_X[index]) > 2 and not textIsANum(dict_X[index]):
                dict_topic[cluster_k] += [dict_X[index]]
                print ' - %s'% dict_X[index]

    return dict_topic

def doSomAndDryTopics(input_list, w2v_model, som_model, clustering_model):
    cleaned_tweet_list = clean_text_ms.cleanText(input_list)
    embedded_words, dict_index2word, dict_word2index = word2vec_ms.getEmbeddedWords(
        cleaned_tweet_list, w2v_model)

    data2unit, data2cell, data2dist, data2saliency, data2saliency_index, data2maps = som_model.predict(
        embedded_words)

    log.info("fit cluster...")

    codebook2cluster = clustering_model.predict(som_model.W)

    topics = getTopics(som_model, embedded_words, dict_index2word)
    save_obj( stopwordsDictFromFile(conf.ConfigSectionMap('STOPWORDS_FILES')),
                  conf.get('MAIN', 'path_pickle_stopwords_dict'))

    dried_topics = dryTopics(topics, codebook2cluster, embedded_words, dict_word2index, dict_index2word, 1, conf)
    return dried_topics

import core.configurations
import numpy, pandas as pd
from gensim.models.word2vec import Word2Vec
from core.utility.serializator import *
from core.micro_services import word2vec_ms
conf = core.configurations.get_conf()

from core.utility.barCharsDocSimilarity import barCharsTopicDoc, barCharsTopicDocs
from core.utility import plot_graph
from core.utility.textIterable2WordsSentencesIterable import TextIterable2WordsSentencesIterable
from core.micro_services import word2vec_ms, clean_text_ms

def getDocumentsMatrix_words(doc_words,  X, dict_words):

    matrix_list = []
    words_list = []
    for word in doc_words:
        if dict_words.has_key(word):
            words_list.append(word)
            matrix_list.append(X[dict_words[word], :])
    num_dimension = len(X[dict_words.values()[0], :])
    return numpy.array(matrix_list).reshape(-1, num_dimension), words_list

def getDriedTopicMatrix(topic_index, dried_topics, X, dict_words):
    words_list = []
    matrix_list = []
    for words_list_topic in dried_topics[topic_index]:
        for word in words_list_topic:
            if dict_words.has_key(word):
                words_list.append(word)
                matrix_list.append(X[dict_words[word], :])

    return numpy.array(matrix_list), words_list

def frequencyDoc(clusters_doc, numTopics):

    frequency_dict = numpy.zeros(numTopics)
    num_elements = len(clusters_doc.tolist())
    for cluster in clusters_doc.tolist():
        frequency_dict[cluster] += 1.0/num_elements

    return frequency_dict

def predictTopics(input_list, w2v_model, som_model, cluster_model, dried_topics, type_chart = "d3"):
    codebook2cluster = cluster_model.predict(som_model.W)

    cleaned_tweet_list = clean_text_ms.cleanText(input_list)
    embedded_words, dict_index2word, dict_word2index = word2vec_ms.getEmbeddedWords(
        cleaned_tweet_list, w2v_model)


    graphs = []
    for index in xrange(codebook2cluster.max() + 1):
        M, words_list = getDriedTopicMatrix(index, dried_topics, embedded_words, dict_word2index)
        if len(words_list) > 10:
            # file_name_index = './data/output/dried_' + str(index) + '.json'
            file_name = conf.get('MAIN', 'MST_dried_topics_d3_base_file')
            file_name_index = file_name + str(index) + '.html'
            graph = plot_graph.plot_similarity_graph(M, words_list, file_name_index, type_chart)
            graphs.append(graph)
            print file_name_index

    return graphs