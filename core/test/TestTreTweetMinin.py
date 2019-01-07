import unittest
import numpy as np
import re
from collections import OrderedDict

def countEmptyList(data):
    count = 0
    for i in data:
        if (i == []):
            count += 1
    return count

# count number of emty codebook
def countEmptyCodebook(data, w, h):
    # create vector of zeros with lenght wxh
    vector = np.zeros(w*h)

    # add +1 for each element x of data in the posizion x of the vector
    for x in data:
        vector[x] = vector[x]+1

    # count zeros
    nr_zeros = w*h-np.count_nonzero(vector)
    return nr_zeros

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
    return dict_codebook2indexes

# create a dictionary that contains all the codebooks and the index of the most representative word (the nearest)
def getMostRepresentativeWord(codebook2indexes, data2distance):

    dict_codebook2most_repr_index = OrderedDict({})
    #for each codebook
    for codebook, indexes in codebook2indexes.iteritems():
        min = data2distance[indexes[0]]
        min_pos = 0
        # get the nearest word
        for i in indexes:
            if (data2distance[i] <= min ):
                min_pos = i
                min = data2distance[i]
        dict_codebook2most_repr_index[codebook] = min_pos
    return dict_codebook2most_repr_index


# check if mySom.data2unit has or not too many empty element
def isGoodResult(data, w, h, threshold=1):
    return countEmptyCodebook(data, w, h) <= threshold

def convertSentenceToListOfWords(sentence):
    return re.sub("[^\w]", " ", sentence).split()

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
            cluster2most_repr_word_index[cluster].append(codebook2most_repr_index[codebook])
    return cluster2most_repr_word_index


def concatWords(list_words, separator = " "):
    sentence = separator.join(list_words)
    return sentence

def meanOfVectors(vectors):
    first_time = True
    for v in vectors:
        if first_time:
            mean = np.array(v)
            first_time = False
        else:
            mean = mean + np.array(v)
    mean = mean / len(vectors)
    return mean.tolist()

# return most represetnative words associated to cluster (separated by -). return an ordered dict with keys = cluster, values = list of most representative words separated by -
def getCluster2mostRepresentativeWords(cluster2most_repr_word_index, dict_index2word):
    cluster2most_repr_words = OrderedDict()
    for cluster, indexes in cluster2most_repr_word_index.iteritems():
        cluster2most_repr_words[cluster] = []
    for cluster, most_repr_word_indexes in cluster2most_repr_word_index.iteritems():
        most_repr_words = {key: value for key, value in dict_index2word.items() if key in most_repr_word_indexes}
        words = concatWords(most_repr_words.values(), separator=" - ")
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
        cluster2mean_vector[cluster] = meanOfVectors(vectors)
    return cluster2mean_vector

def prova():
    print "a"

class TestStringMethods(unittest.TestCase):
    def testGetCluster2meanVector(self):
        cluster2most_repr_word_index = OrderedDict(
            [(0, [0,1]),
             (1, [2,3])])
        embedded_words = [[1,2],[3,4],[0,2],[8,2]]
        result = OrderedDict(
            [(0, [2,3]),
             (1, [4,2])])
        self.assertEqual(getCluster2meanVector(cluster2most_repr_word_index, embedded_words), result)

    def testGetCluster2mostRepresentativeWords(self):
        cluster2most_repr_word_index = OrderedDict(
            [(0, [0,2]),
             (1, [3,1]),
             (2, [4])])
        dict_index2word = OrderedDict(
            [(0, 'mela'),
             (1, 'pera'),
             (2, 'banana'),
             (3, 'mora'),
             (4, 'limone')])
        result = OrderedDict(
            [(0, 'mela - banana'),
             (1, 'pera - mora'),
             (2, 'limone')])

        self.assertEqual(getCluster2mostRepresentativeWords(cluster2most_repr_word_index, dict_index2word),  result)

    def testMeanOfVectors(self):
        list = [[0, 5],[6, 3]]
        self.assertEqual(meanOfVectors(list), [3, 4])

    def testConcatWords(self):
        list_words = ['ciao', 'sono', 'ioooooooooooo']
        self.assertEqual(concatWords(list_words, separator=" - "), 'ciao - sono - ioooooooooooo')

    def testGetCluster2mostRepresentativeWordIndex(self):
        cluster2codebook = OrderedDict(
            [(0, [0,2]),
             (1, [3,1]),
             (2, [5])]
        )
        codebook2most_repr_index = OrderedDict(
            [(0, 3),
             (1, 2),
             (2, 1),
             (3, 4),
             (4, 5),
             (5, 0)]
        )
        cluster2word = OrderedDict(
            [(0, [3,1]),
             (1, [4,2]),
             (2, [0])]
        )
        self.assertEqual(getCluster2mostRepresentativeWordIndex(cluster2codebook, codebook2most_repr_index), cluster2word)

    def testGetCluster2codebook(self):
        data2cluster = [1,0,1,2,0]
        data2unit = [0,3,2,5,1]
        self.assertEqual(getCluster2codebook(data2cluster, data2unit)[1], [0,2])
        self.assertEqual(getCluster2codebook(data2cluster, data2unit)[0], [3,1])
        self.assertEqual(getCluster2codebook(data2cluster, data2unit)[2], [5])

    def testCountEmptyList1(self):
        data = np.array([[1, 2, 3], [], [4, 5, 6], []])
        self.assertEqual(countEmptyList(data), 2)

    def testCountEmptyList2(self):
        data = np.array([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(countEmptyList(data), 0)

    def testCountEmptyCodebook(self):
        data = np.array([1, 2])
        w = h = 2
        self.assertEqual(countEmptyCodebook(data, w, h), 2)

    def testIsGood0(self):
        data = np.array([1, 2])
        w = h = 2
        self.assertTrue(isGoodResult(data, w, h, 2))

    def testIsGood1(self):
        data = np.array([1, 2])
        w = h = 2
        self.assertFalse(isGoodResult(data, w, h))

    def testIsGood2(self):
        data = np.array([1, 2, 3])
        w = h = 2
        self.assertTrue(isGoodResult(data, w, h))

    def testGetCodebook2indexes(self):
        data2unit = [1,0,1,2,0]
        dict = OrderedDict(
                    [(0, [1,4]),
                     (1, [0,2]),
                     (2, [3])]
        )
        self.assertEqual(getCodebook2indexes(data2unit)[0], dict[0])
        self.assertEqual(getCodebook2indexes(data2unit)[1], dict[1])
        self.assertEqual(getCodebook2indexes(data2unit)[2], dict[2])

    def testGetMostRepresentativeWord(self):
        data2unit = [1,0,1,2,0]
        codebook2indexes = getCodebook2indexes(data2unit)
        distances = [1, 5, 3, 2, 1]
        dict = OrderedDict(
                    [(0, 4),
                     (1, 0),
                     (2, 3)]
        )
        self.assertEqual(getMostRepresentativeWord(codebook2indexes, distances)[0], dict[0])
        self.assertEqual(getMostRepresentativeWord(codebook2indexes, distances)[1], dict[1])
        self.assertEqual(getMostRepresentativeWord(codebook2indexes, distances)[2], dict[2])

if __name__ == '__main__':
    unittest.main()
    # prova()
