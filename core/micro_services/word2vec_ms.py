from gensim.models.word2vec import Word2Vec
from collections import OrderedDict
import numpy
import logging

import core.utility.logger as logger
import core.utility.utilities as utilities
import core.configurations
import os
from core.micro_services import clean_text_ms
from core.utility.serializator import save_obj, load_obj

conf = core.configurations.get_conf()
log_path = conf.get('MAIN', 'log_path')
log_file_name = conf.get('MAIN', 'log_file_name')
log = logger.getLogger(__name__, log_path , log_file_name)


class Word2VecMS:

    def __init__(self, tweets, w2v_model=None):
        # load trained model W2V
        if (w2v_model==None):
            self.model = Word2Vec.load(conf.get('MAIN', 'path_pickle_w2v_model'))
        else:
            self.model = w2v_model

        self.vec2tweets = {}
        self.vec2word = {}
        self.word2tweet = {}

        self.tweets = tweets
        self.cleaned_tweets = clean_text_ms.cleanText(tweets)

        if os.path.exists(conf.get('MAIN', 'path_vec2tweets')):
            self.vec2tweets = load_obj(conf.get('MAIN', 'path_vec2tweets'))
        if os.path.exists(conf.get('MAIN', 'path_vec2word')):
            self.vec2word = load_obj(conf.get('MAIN', 'path_vec2word'))
        if os.path.exists(conf.get('MAIN', 'path_word2tweet')):
            self.word2tweet = load_obj(conf.get('MAIN', 'path_word2tweet'))

        self.embedded_words, self.index2word, self.word2index = getEmbeddedWords(self.cleaned_tweets, w2v_model)


    def computeWord2Tweets(self):
        """
        computes dictionary for each vector the list of the tweets related
        """
        self.vec2tweets = {}
        self.vec2word = {}
        self.word2tweet = {}

        for key in self.model.wv.vocab.keys():
            vec = self.model[key]
            self.vec2tweets[utilities.convertListOfNumToString(vec)] = []
            self.word2tweet[key] = []

        for i in range(len(self.cleaned_tweets)):
            for word in utilities.convertSentenceToListOfWords(self.cleaned_tweets[i]):
                try:
                    vec = self.model.wv[word]
                    self.vec2tweets[utilities.convertListOfNumToString(vec)].append(self.tweets[i])
                    self.vec2word[utilities.convertListOfNumToString(vec)] = word
                    self.word2tweet[word].append(self.tweets[i])
                except:
                    pass

    def saveObject(self):
        self.model.save(conf.get('MAIN', 'path_pickle_w2v_model'))
        save_obj(self.vec2tweets, conf.get('MAIN', 'path_vec2tweets'))
        save_obj(self.vec2word, conf.get('MAIN', 'path_vec2word'))
        save_obj(self.word2tweet, conf.get('MAIN', 'path_word2tweet'))

    def getVec2word(self, vec):
        self.vec2word[utilities.convertListOfNumToString(vec)]

    def getVec2tweets(self, vec):
        self.vec2tweets[utilities.convertListOfNumToString(vec)]


def trainNewModelW2Vmodel(corpus, identifier):
    #save temporary file to save model training status
    filename = conf.get("MAIN", "path_pickle_w2v_model_incr_fold")+"word2vec_"+str(identifier)+"_training.txt"
    text_file = open(filename, "w")
    text_file.close()

    trainW2Vmodel(corpus, new_model=True, identifier=identifier)



def trainW2Vmodel(corpus, new_model=False, identifier=""):
    log.info("Training W2V Model")

    # covert list of sentences to list of words
    list_corpus = []
    for text in corpus:
        if (str(text) != "nan"):
            list_corpus.append(utilities.convertSentenceToListOfWords(text))
    corpus = list_corpus

    # count num of words
    flat_list = [item for sublist in corpus for item in sublist]
    num_words = len(flat_list)

    debugging = conf.get('MAIN', 'debugging')

    if (debugging == 'True'):
        n_epoch = 10000
    else:
        n_epoch = int(10 ** 9 / num_words)

    #epochs = iter
    model = Word2Vec(size           =   conf.getint('W2V', 'size'),
                     min_count      =   conf.getint('W2V', 'min_count'),
                     sg             =   conf.getint('W2V', 'sg'),
                     window         =   conf.getint('W2V', 'window'),
                     iter           =   n_epoch,
                     alpha          =   conf.getfloat('W2V', 'alpha'),
                     workers        =   conf.getint('W2V', 'workers')
                    )

    model.build_vocab(corpus, progress_per=conf.getint('W2V', 'progress_per'))
    #Train the model over train_reviews (this may take several minutes)

    model.train(corpus, total_examples=model.corpus_count, epochs=model.iter)

    if (new_model == False):
        model.save(conf.get('MAIN', 'path_pickle_w2v_model'))
        log.info("Model trained")
    else:
        filename = conf.get('MAIN', 'path_pickle_w2v_model_incr_fold')+"word2vec_"+str(identifier)+".pickle"
        model.save(filename)

def collectWords(model):

    dict_X = OrderedDict({})
    dict_words = OrderedDict({})
    row_index=0
    list_X = []

    for key in model.wv.vocab.keys():
        dict_X[row_index] = key.lower()
        dict_words[ key.lower()] = row_index
        list_X += [model[key]]
        row_index += 1

    X = numpy.array(list_X)

    return X, dict_X, dict_words
    # X vector matrix
    # dict_X {1: 'internet'}
    # dict_words {'internet': 1}

# get embedded words of tweet and dictionaries of words and indices
# words are note repeted
def getEmbeddedWords(sentences, model=None):

    log.info("getting Embedded Words from input")

    # load trained model W2V
    if (model is None):
        model = Word2Vec.load(conf.get('MAIN', 'path_pickle_w2v_model'))

    list_of_sentences = []
    embedded_words = []
    dict_index2word_tweet = OrderedDict({})
    dict_word2index_tweet = OrderedDict({})
    row_index = 0

    # from list of sentence to list of list of words
    for text in sentences:
        if (str(text) != "nan"):
            list_of_sentences.append(utilities.convertSentenceToListOfWords(text))

    for sentence in list_of_sentences:
        for word in sentence:
            try:
                # if word is not yet inserted
                if word not in dict_word2index_tweet:
                    embedded_words.append(model.wv[word])
                    dict_index2word_tweet[row_index] = word
                    dict_word2index_tweet[word] = row_index
                    row_index += 1
            # we are here if w2v model does not have the word
            except KeyError:
                pass

    return numpy.array(embedded_words), dict_index2word_tweet, dict_word2index_tweet

def test():
    import os
    os.chdir("../..")

    tweets = ["@Tre_It l'unico operatore che ti da una tariffa PER SEMPRE e poi cambia tariffa ogni due per tre...Truffatori",
"@potina83 @Tre_It si credo mantengano la cosa anche in UK , tanto ancora manca parecchio prima che la Brexit sia co https: / /t.co /YCmZMVOntL",
"@Tre_It Salve,posso andare in un 3store e attivare la super internet ten limited edition avendo io 16 anni? Serve un maggiorenne per la sim?"]

    word2VecMS = Word2VecMS(tweets)

    word2VecMS.computeWord2Tweets()

    for vec in word2VecMS.vec2word:
        print word2VecMS.vec2word[vec]
        print word2VecMS.vec2tweets[vec]

if __name__ == '__main__':
    test()