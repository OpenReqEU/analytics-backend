import codecs
import pylab
import pandas as pd
from math import *

# custom
from core.micro_services.word2vec_ms import *
import core.configurations
from core.utility.serializator import load_obj, save_obj
import word2vec_ms
import clean_text_ms
from core.utility import plot_graph
from core.utility.utilities import *
from core.AdvancedSOM import ASOM
from word2vec_ms import Word2VecMS

conf = core.configurations.get_conf()
log_path = conf.get('MAIN', 'log_path')
log_file_name = conf.get('MAIN', 'log_file_name')
log = logger.getLogger(__name__, log_path, log_file_name)


class SOM_MS:
    def __init__(self, input_list, w2v_model=None, som_model=None):
        if (som_model == None):
            self.som = load_obj(conf.get('MAIN', 'path_pickle_som_model'))
        else:
            self.som = som_model
        self.data2unit = []
        self.data2cell = []
        self.data2dist = []
        self.data2saliency = []
        self.data2saliency_index = []
        self.data2maps = []

        self.codebook2indexes = {}
        if (som_model == None):
            self.word2vecMS = Word2VecMS(input_list)
        else:
            self.word2vecMS = Word2VecMS(input_list, w2v_model)

    def predict(self, embedded_words):
        self.data2unit, self.data2cell, self.data2dist, self.data2saliency, self.data2saliency_index, self.data2maps = self.som.predict(
            embedded_words)
        self.codebook2indexes = getCodebook2indexes(self.data2unit)
        self.codebook2words, self.codebook2index = getCodebook2Word(self.data2unit, self.data2dist,
                                                                    self.word2vecMS.index2word)

    def getCodebookContent(self, embedded_words, index):
        return embedded_words[self.data2unit == index]

    def getCodebookTweets(self, i):
        all_tweets = []
        indexes = self.codebook2indexes[i]

        for index in indexes:
            word = self.word2vecMS.index2word[index]
            tweets = self.word2vecMS.word2tweet[word]
            for tweet in tweets:
                all_tweets.append(tweet)

        return all_tweets

def trainNewModelBestSom(w2v_model, identifier):
    #save temporary file to save model training status
    filename = conf.get('MAIN', 'path_pickle_som_model_incr_fold') + "som_" + str(identifier) +"_training.txt"
    text_file = open(filename, "w")
    text_file.close()

    trainBestSom(w2v_model, new_model=True, identifier=identifier)

def trainBestSom(w2v_model, new_model=False, identifier=""):
    # get w2v words, dict words and vectors only for tweet
    embedded_words_t_w, dict_index2word_t_w, dict_word2indext_w = collectWords(w2v_model)

    width = int(conf.get('ADVANCED_ASOM', 'width'))
    height = int(conf.get('ADVANCED_ASOM', 'height'))
    empty_codebook_threshold = int(conf.getboolean('ADVANCED_ASOM', 'empty_codebook_threshold'))

    log.info("training som [" + str(width) + "x" + str(height) + "]")
    mySom = trainSOM(embedded_words_t_w, dict_index2word_t_w, conf, width, height)

    min_size_codebook_mtx = int(conf.get('ADVANCED_ASOM', 'min_size_codebook_mtx'))
    step_codebook_mtx = int(conf.get('ADVANCED_ASOM', 'step_codebook_mtx'))

    # decrease som dimensions if we have more than one codebook empty
    while (not isGoodResult(mySom, width, height,
                            empty_codebook_threshold) and width > min_size_codebook_mtx + step_codebook_mtx):
        log.info("training som [" + str(width) + "x" + str(height) + "]")
        width = height = height - 2
        mySom = trainSOM(embedded_words_t_w, dict_index2word_t_w, conf, width, height)

    if (new_model == False):
        save_obj(mySom, conf.get('MAIN', 'path_pickle_som_model'))
        log.info("Model trained")
        mySom = load_obj(conf.get('MAIN', 'path_pickle_som_model'))
    else:
        filename = conf.get('MAIN', 'path_pickle_som_model_incr_fold') + "som_" + str(identifier) + ".pickle"
        save_obj(mySom, filename)

def trainNewModelCodebookCluster(som_model, identifier):
    #save temporary file to save model training status
    filename = conf.get('MAIN', 'path_pickle_codebook_cluster_model_incr_fold') + "codebook_cluster_" + str(
        identifier) +"_training.txt"
    text_file = open(filename, "w")
    text_file.close()

    trainCodebookCluster(som_model, new_model=True, identifier=identifier)

def trainCodebookCluster(som_model, new_model=False, identifier=""):
    make_figure = False
    cluster_model = som_model.fit_cluster(cluster_model         =   None,
                      perc_subsampling=0.,
                      default_cluster_model=0,
                      num_cluster_min       =   conf.getint('ADVANCED_ASOM', 'num_cluster_min'),
                      num_cluster_max       =   conf.getint('ADVANCED_ASOM', 'num_cluster_max'))

    if (new_model == False):
        save_obj(som_model.cluster_model, conf.get('MAIN', 'path_pickle_codebook_cluster_model'))
        log.info("saved cluster model in " + conf.get('MAIN', 'path_pickle_codebook_cluster_model'))
    else:
        filename = conf.get('MAIN', 'path_pickle_codebook_cluster_model_incr_fold') + "codebook_cluster_" + str(identifier) + ".pickle"
        save_obj(som_model.cluster_model, filename)
        log.info("saved cluster model in " + filename)


# count number of emty codebook
def countEmptyCodebook(data, w, h):
    # create vector of zeros with lenght wxh
    vector = np.zeros(w * h)

    # add +1 for each element x of data in the posizion x of the vector
    for x in data:
        vector[x] = vector[x] + 1

    # count zeros
    nr_zeros = w * h - np.count_nonzero(vector)
    return nr_zeros


# check if mySom.data2unit has or not too many empty element
def isGoodResult(som, w, h, threshold=1):
    return countEmptyCodebook(som.data2unit, w, h) <= threshold


# load trained model SOM, make the predict and plot the result (predict and plot)
def doSomAndPlot(som, embedded_words, dict_index2word, file_name, type_chart):
    # mySom.data2unit says for each words in which codebook is contained
    data2unit, data2cell, data2dist, data2saliency, data2saliency_index, data2maps = som.predict(
        embedded_words)

    # --------- OUTPUT: get topics: creating dictionary codebook, words ----------------
    dict_codebook_topic = getTopic(data2unit, dict_index2word)

    path = './data/output/SOM_output.txt'
    printTable(dict_codebook_topic, path)
    log.info("matrix codebook - words [" + str(som.width) + "x" + str(som.height) + "]")

    # -------------------------OUTPUT Miminum Spanning Tree-------------------------------
    codebook2words, codebook2index = getCodebook2Word(data2unit, data2dist, dict_index2word)

    # most representative vectors
    matrix_list = []
    for index in codebook2index.values():
        matrix_list.append(embedded_words[index, :])

    # build MST
    url = plot_graph.plot_similarity_graph(numpy.array(matrix_list), codebook2words.values(), file_name, type_chart)

    return url


# load trained model SOM, make the predict and return each word for each codebook
def getCodebookWords(tweet_rows, w2v_model, som_model):
    log.info("predict and plot")
    cleaned_tweet_list = clean_text_ms.cleanText(tweet_rows)

    # get embedded words from input
    embedded_words_tweets, dict_index2word_tweet, dict_word2index_tweet = word2vec_ms.getEmbeddedWords(
        cleaned_tweet_list, w2v_model)

    # mySom.data2unit says for each words in which codebook is contained
    data2unit, data2cell, data2dist, data2saliency, data2saliency_index, data2maps = som_model.predict(
        embedded_words_tweets)

    dict_codebook_topic = getTopic(data2unit, dict_index2word_tweet)
    return dict_codebook_topic


# load cluster model and plot results
def doClusteringAndPlot(tweet_rows, file_name):
    log.info("clustering and plot")
    # clean input
    cleaned_tweet_list = clean_text_ms.cleanText(tweet_rows)

    # get embedded words from input
    embedded_words_tweets, dict_index2word_tweet, dict_word2index_tweet = word2vec_ms.getEmbeddedWords(
        cleaned_tweet_list)

    # load SOM and cluster model
    mySom = load_obj(conf.get('MAIN', 'path_pickle_som_model'))
    cluster_model = load_obj(conf.get('MAIN', 'path_pickle_cluster_model'))
    log.info("SOM model loaded " + conf.get('MAIN', 'path_pickle_som_model'))

    # mySom.data2unit says for each words in which codebook is contained
    data2unit, data2cell, data2dist, data2saliency, data2saliency_index, data2maps = mySom.predict(
        embedded_words_tweets)

    log.info("fit cluster...")
    # make clustering
    data2cluster = cluster_model.predict(embedded_words_tweets)
    # data2cluster = cluster_model.predict(embedded_words_tweets)

    # -------------------------OUTPUT print table of clusters------------------------------
    path = './data/output/cluster_output.txt'
    dict_cluster_topic = getTopic(data2cluster, dict_index2word_tweet)
    printTable(dict_cluster_topic, path)

    # -------------------------OUTPUT bubble-chart cluster-----------------------
    codebook2word, codebook2index = getCodebook2Word(data2unit, data2dist, dict_index2word_tweet)
    dict_cluster2codebook = getCluster2codebook(data2cluster, data2unit)
    cluster2most_repr_word_index = getCluster2mostRepresentativeWordIndex(dict_cluster2codebook,
                                                                          codebook2index.values())

    # dict cluster - most represetative words
    cluster2most_repr_words = getCluster2mostRepresentativeWords(cluster2most_repr_word_index, dict_index2word_tweet)
    # dict cluster - mean vector of most representative vectors
    cluster2mean_vector = getCluster2meanVector(cluster2most_repr_word_index, embedded_words_tweets)

    cell_frequency = mySom.cellFrequencyDistribution(embedded_words_tweets)

    # save_obj(data2cluster, "./data2cluster.pickle")
    # save_obj(codebook2word, "./codebook2word.pickle")
    # save_obj(dict_word2index_tweet, "./dict_word2index_tweet.pickle")
    # save_obj(cell_frequency, "./cell_frequency.pickle")
    url = buildClusterCsv(data2cluster, codebook2word, dict_word2index_tweet, cell_frequency, file_name)

    # build MST
    # url = plot_graph.plot_similarity_graph(numpy.array(cluster2mean_vector.values()),
    #                                        cluster2most_repr_words.values(), file_name, conf, "markers", type_chart)
    return url


# building csv for bubble chart d3 viewin clustering and frequencies
# each color is a cluster
# the ball is as large as the frequency
def buildClusterCsv(data2cluster, codebook2word, dict_word2index_tweet, cell_frequency, file_name):
    # BUILD DATA2FREQUENCIES
    a = np.array(cell_frequency)
    a = a.transpose()
    unit2frequencies = [item for sublist in cell_frequency for item in sublist]
    # removing 0.0 frequencies
    unit2frequencies = filter(lambda a: a != 0.0, unit2frequencies)

    codebook2cluster = {}
    # BUILD CODEBOOK2CLUSTER
    for c, w in codebook2word.iteritems():
        index = dict_word2index_tweet[w]
        cluster = data2cluster[index]
        codebook2cluster[c] = cluster

    list_words = []
    for c, w in codebook2cluster.iteritems():
        list_words.append(str(abs(codebook2cluster[c])) + "." + codebook2word[c])
        # list_words.append(str(codebook2cluster[c])+"."+codebook2word[c])

    new_freq = [i * 1000 * 1000 for i in unit2frequencies]
    d = dict(id=np.array(list_words), value=np.array(new_freq))
    df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in d.iteritems()]))
    df.to_csv(file_name, index=False)

    log.info("Clustering saved in " + file_name)
    return file_name


def getCodebookActivation(som_model=None):
    # load SOM
    if (som_model == None):
        som_model = load_obj(conf.get('MAIN', 'path_pickle_som_model'))

    som_model.plot_activations()
    filename = conf.get('MAIN', 'codebook_activation_filename')
    pylab.savefig(filename)
    return filename


def getUmatrix(som_model=None):
    # load SOM
    if (som_model == None):
        som_model = load_obj(conf.get('MAIN', 'path_pickle_som_model'))
    UM, unit_xy = som_model.evaluate_UMatrix()
    filename = conf.get('MAIN', 'umatrix_filename')
    plot_graph.plotMatrix(UM, filename)
    return filename


# num_of_topic number of topic to visualize in the plot, if 0 -> all
def getCellFrequencyDistribution(tweet_rows, w2v_model, mySom, num_of_topic=0, type_chart='bar'):
    cleaned_tweet_list = clean_text_ms.cleanText(tweet_rows)

    # get embedded words from input
    embedded_words_tweets, dict_index2word_tweet, dict_word2index_tweet = word2vec_ms.getEmbeddedWords(
        cleaned_tweet_list, w2v_model)

    # predict
    data2unit, data2cell, data2dist, data2saliency, data2saliency_index, data2maps = mySom.predict(
        embedded_words_tweets)

    cell_frequency = mySom.cellFrequencyDistribution(embedded_words_tweets)

    codebook2word, codebook2index = getCodebook2Word(data2unit, data2dist, dict_index2word_tweet)

    if (type_chart == 'bubble'):
        file_name = conf.get('MAIN', 'topic_frequencies_file_bubble')
    else:
        file_name = conf.get('MAIN', 'topic_frequencies_file_bar')

    # adding empty word for empty codebook
    for i in range(0, mySom.width * mySom.width):
        if i not in codebook2word.keys():
            codebook2word[i] = ''
    codebook2word = OrderedDict(sorted(codebook2word.items()))

    url = plot_graph.plotFrequencyGraph(cell_frequency, codebook2word.values(), file_name, num_of_topic, type_chart)

    log.info("Frequency bubble chart in " + file_name)

    f = codecs.open(file_name, 'r')
    html = f.read()
    return html


def getCostOfSom(som_model=None):
    # load SOM
    if (som_model == None):
        som_model = load_obj(conf.get('MAIN', 'path_pickle_som_model'))

    cost = som_model.estimate_cost2(som_model.X[0:10000])
    cost = cost * 100
    cost = round(cost, 2)
    cost = str(cost) + " %"

    log.info("cost: " + cost)
    return cost


def getCodebook2Word(data2unit, data2dist, dict_index2word_tweet):
    # dict_codebook2indexes = getCodebook2indexes(data2unit)
    # dict_codebook2most_repr_index = getMostRepresentativeWord(dict_codebook2indexes, data2dist)
    # most_repr_indexes = dict_codebook2most_repr_index.values()
    # most_repr_words = {key: value for key, value in dict_index2word_tweet.items() if key in most_repr_indexes}
    # return most_repr_words

    dict_codebook2indexes = getCodebook2indexes(data2unit)
    dict_codebook2most_repr_index = getMostRepresentativeWord(dict_codebook2indexes, data2dist)

    dict_codebook2word = {}
    for codebook, index in dict_codebook2most_repr_index.iteritems():
        dict_codebook2word[codebook] = dict_index2word_tweet[index]

    return OrderedDict((sorted(dict_codebook2word.items()))), dict_codebook2most_repr_index
    # return dict_codebook2word, dict_codebook2most_repr_index


# create dictionary of topic : keys = codebook values = words
def getTopic(data2unit, dict_index2word):
    # create dictionary of codebook
    dict_codebook_topic = OrderedDict({})
    for i in range(0, len(data2unit)):
        dict_codebook_topic[data2unit[i]] = []
    # fill dictionary with words
    for i in range(0, len(data2unit)):
        codebook = dict_codebook_topic[data2unit[i]]
        codebook.append(dict_index2word[i])
    return dict_codebook_topic


# Printer
def printTable(data, path):
    f1 = open(path, 'w+')

    # Column width
    AttrColLen = 25
    ValueColLen = 15
    colwidth1 = "{0:<" + str(AttrColLen) + "}"
    colwidth2 = "{0:<" + str(ValueColLen) + "}"

    values = [1]
    for key, values in sorted(data.items()):
        print >> f1, "|" + AttrColLen * "-" + ((ValueColLen + 2) * len(values)) * "-" + "-|"
        print >> f1, "| " + colwidth1.format(key) + "|",
        for i in xrange(len(values)):
            print >> f1, colwidth2.format(values[i]) + "|",
        print >> f1, ""
    print >> f1, "|" + AttrColLen * "-" + ((ValueColLen + 2) * len(values)) * "-" + "-|"


# create dictionary codebook/indexes words : keys = codebook, values = indexes of words
def getCodebook2indexes(data2unit):
    # create dictionary of codebook
    dict_codebook2indexes = OrderedDict({})
    for i in range(0, len(data2unit)):
        dict_codebook2indexes[data2unit[i]] = []
    # fill dictionary with indexes
    for i in range(0, len(data2unit)):
        codebook = dict_codebook2indexes[data2unit[i]]
        codebook.append(i)
    return OrderedDict((sorted(dict_codebook2indexes.items())))


# create a dictionary that contains all the codebooks and the index of the most representative word (the nearest)
def getMostRepresentativeWord(codebook2indexes, data2distance):
    dict_codebook2most_repr_index = OrderedDict({})
    # for each codebook
    for codebook, indexes in codebook2indexes.iteritems():
        min = data2distance[indexes[0]]
        min_pos = 0
        # get the nearest word
        for i in indexes:
            if (data2distance[i] <= min):
                min_pos = i
                min = data2distance[i]
        dict_codebook2most_repr_index[codebook] = min_pos
    return dict_codebook2most_repr_index


# return codebook associated to cluster. return a ordered dict with keys = cluster, values = list of codebook
def getCluster2codebook(data2cluster, data2unit):
    cluster2codebook = OrderedDict()
    for i in data2cluster:
        cluster2codebook[i] = []
    for i in range(0, len(data2cluster)):
        nr_cluster = data2cluster[i]
        cluster = cluster2codebook[nr_cluster]
        value = data2unit[nr_cluster]
        cluster.append(data2unit[i])
    return cluster2codebook


# return most represetnative word index associated to cluster. return an ordered dict with keys = cluster, values = list of most representative word indexes
def getCluster2mostRepresentativeWordIndex(cluster2codebook, codebook2most_repr_index):
    cluster2most_repr_word_index = OrderedDict()
    for cluster, codebooks in cluster2codebook.iteritems():
        cluster2most_repr_word_index[cluster] = []
    for cluster, codebooks in cluster2codebook.iteritems():
        for codebook in codebooks:
            try:
                cluster2most_repr_word_index[cluster].append(codebook2most_repr_index[codebook])
            except:
                pass
    return cluster2most_repr_word_index


# return most represetnative words associated to cluster (separated by -). return an ordered dict with keys = cluster, values = list of most representative words separated by -
def getCluster2mostRepresentativeWords(cluster2most_repr_word_index, dict_index2word):
    cluster2most_repr_words = OrderedDict()
    for cluster, indexes in cluster2most_repr_word_index.iteritems():
        cluster2most_repr_words[cluster] = []
    for cluster, most_repr_word_indexes in cluster2most_repr_word_index.iteritems():
        most_repr_words = {key: value for key, value in dict_index2word.items() if key in most_repr_word_indexes}
        most_repr_words = OrderedDict((sorted(most_repr_words.items())))
        separator = "<br>"
        words = concatWords(most_repr_words.values(), separator=separator)
        cluster2most_repr_words[cluster] = words
    return cluster2most_repr_words


# return most represetnative words associated to cluster (separated by -). return an ordered dict with keys = cluster, values = list of most representative words separated by -
def getCluster2meanVector(cluster2most_repr_word_index, embedded_words):
    cluster2mean_vector = OrderedDict()
    for cluster, indexes in cluster2most_repr_word_index.iteritems():
        cluster2mean_vector[cluster] = []
    for cluster, most_repr_word_indexes in cluster2most_repr_word_index.iteritems():
        vectors = []
        for index in most_repr_word_indexes:
            vectors.append(embedded_words[index])
        words = meanOfVectors(vectors)
        if (words != None):
            cluster2mean_vector[cluster] = words
    return cluster2mean_vector


def trainSOM(X, dict_X, conf, width=20, height=20):
    som = ASOM(alpha_max=conf.getfloat('ADVANCED_ASOM', 'alpha_max'),
               alpha_min=conf.getfloat('ADVANCED_ASOM', 'alpha_min'),
               height=height,
               width=width,
               outlier_unit_threshold=conf.getfloat('ADVANCED_ASOM', 'outlier_unit_threshold'),
               outlier_percentile=conf.getfloat('ADVANCED_ASOM', 'outlier_percentile'),
               Koutlier_percentile=conf.getint('ADVANCED_ASOM', 'Koutlier_percentile'),
               learning_rate_percentile=conf.getfloat('ADVANCED_ASOM', 'learning_rate_percentile'),
               distance_metric="cosine"
               )

    num_epoch = conf.getint('ADVANCED_ASOM', 'num_epoch')
    # DEBUGGING
    debugging = conf.get('MAIN', 'debugging')
    if (debugging == 'True'):
        num_epoch = 10
    som.train_batch(X,
                    num_epoch=num_epoch,
                    training_type=conf.get('ADVANCED_ASOM', 'training_type'),
                    batch_size=None,  # int(conf.get('ADVANCED_ASOM', 'batch_size')),
                    fast_training=conf.getboolean('ADVANCED_ASOM', 'fast_training'),
                    verbose=conf.getint('ADVANCED_ASOM', 'verbose')
                    )

    return som


def getSomWithPrediction(input_list, w2v_model=None, som_model=None):
    # clean input
    cleaned_input_list = clean_text_ms.cleanText(input_list)

    # get embedded words from input
    embedded_words, dict_index2word, dict_word2index = word2vec_ms.getEmbeddedWords(
        cleaned_input_list, w2v_model)
    som_MS = SOM_MS(input_list, w2v_model, som_model)
    som_MS.predict(embedded_words)
    return som_MS


def getCodebooksTweets(input_list, w2v_model=None, som_model=None):
    som_MS = getSomWithPrediction(input_list, w2v_model, som_model)

    n_codebooks = som_MS.som.width * som_MS.som.height

    data1 = {}
    data2 = {}
    data = []

    for i in range(n_codebooks):
        try:
            if (som_MS.codebook2words[i] != u'\xf3'):
                data2["name"] = som_MS.codebook2words[i]
                data2["tweets"] = som_MS.getCodebookTweets(i)
            else:
                data2["name"] = ""
                data2["tweets"] = []
        except:
            data2["name"] = ""
            data2["tweets"] = []
        data1[i] = data2
        data.append(data1)
    return data


def getCodebooksName(input_list, w2v_model=None, som_model=None):
    som_MS = getSomWithPrediction(input_list, w2v_model, som_model)

    n_codebooks = som_MS.som.width * som_MS.som.height

    data = []

    for i in range(n_codebooks):
        data1 = {}
        data2 = {}
        try:
            if (som_MS.codebook2words[i] != u'\xf3'):
                data2["name"] = som_MS.codebook2words[i]
            else:
                data2["name"] = ""
        except:
            data2["name"] = ""
        data1[i] = data2
        data.append(data1)
    return data


def getCodebookTweets(input_list, i, w2v_model=None, som_model=None):
    som_MS = getSomWithPrediction(input_list, w2v_model, som_model)

    data1 = {}
    data2 = {}
    data = []

    try:
        data2["name"] = som_MS.codebook2words[i]
        data2["tweets"] = som_MS.getCodebookTweets(i)
    except Exception as e:
        print e
        data2["name"] = ""
        data2["tweets"] = []
    data1[i] = data2
    data.append(data1)

    # import json
    # json_data = json.dumps(data)
    # return json_data
    return data


# def getCodebookTweetsJson(input_list):
#     data = getCodebookTweets(input_list)
#
#     import json
#     json_data = json.dumps(data)
#     return json_data

def getLocalCodebooksName():
    input_list = readLocalInputList()
    return getCodebooksName(input_list)


def getLocalCodebookTweets(i):
    input_list = readLocalInputList()

    return getCodebookTweets(input_list, i)


def getLocalCodebooksTweets():
    input_list = readLocalInputList()
    return getCodebooksTweets(input_list)


def readLocalInputList():
    document_path_file = conf.get('MAIN', 'path_document')
    input_list = pd.read_csv(document_path_file, encoding='utf-8', error_bad_lines=False)
    input_list1 = input_list[input_list['idriferimento_ricerca'] == 5]
    input_list2 = input_list[input_list['idriferimento_ricerca'] == 6]
    input_list = pd.concat([input_list1, input_list2])

    input_list = input_list['messaggio'].tolist()
    return input_list


if __name__ == '__main__':
    import os

    os.chdir("../..")
    getLocalCodebooksTweets()
