import core.genspacyrank as genspacyrank

from core.utility.serializator import load_obj, save_obj
from core.utility import utilities

import core.configurations
import core.utility.logger as logger

conf = core.configurations.get_conf()
log_path = conf.get('MAIN', 'log_path')
log_file_name = conf.get('MAIN', 'log_file_name')
log = logger.getLogger(__name__, log_path, log_file_name)

from core.micro_services import clean_text_ms
from threading import Lock, Thread

lock = Lock()

def trainingNewModelBigram(input_list, identifier):
    # save temporary file to save model training status
    filename = conf.get('MAIN', 'path_pickle_bigram_model_incr_fold') + "bigram_" + str(
        identifier) + "_training.txt"
    text_file = open(filename, "w")
    text_file.close()

    trainingBigram(input_list, new_model=True, identifier=identifier)

def trainingBigram(input_list, new_model=False, identifier=""):
    lang = 'it_core_news_sm'
    bigram_model, _ = genspacyrank.training_ngram(corpus=input_list, lang=lang, min_count=1, threshold=2,
                                                  max_vocab_size=40000000,
                                                  delimiter='_', progress_per=10000, scoring='default',
                                                  rm_stopwords=True)

    if (new_model == False):
        save_obj(bigram_model, conf.get('MAIN', 'path_pickle_bigram_model'))
    else:
        filename = conf.get('MAIN', 'path_pickle_bigram_model_incr_fold') + "bigram_" + str(
            identifier) + ".pickle"
        save_obj(bigram_model, filename)
        log.info("saved bigram model in " + filename)

def extractKeywords(input_list, bigram_model):
    input_list = clean_text_ms.cleanText(input_list)

    # output = []
    # output = splitText(input_list, output)
    keywds_list = []

    i = 0
    status = 0
    for input in input_list:
        keywds, graph, text, ngrams = genspacyrank.extract_keywords(text=input, lang='en',
                                                                    bigram_model=bigram_model, trigram_model=None,
                                                                    selected_pos=['V', 'N', 'J'], rm_stopwords=True)
        keywords = []
        for k in keywds:
            keywords.append(k[0])
        keywds_list.append(keywords)
        print "Processing message "+str(i)+" of "+str(len(input_list))
        i = i + 1
        # DEBUGGING
        debugging = conf.get('MAIN', 'debugging')
        if (debugging == 'True'):
            if i == 1000:
                return keywds_list

    # keywds = utilities.convertLisfOfListToList(keywds_list)
    # return keywds
    return keywds_list


def splitText(input_list, output, lenght=50000):
    text = unicode(input_list)

    if (len(text) > lenght):
        if len(input_list) == 1:
            raise Exception(
                'A single list lenght is greater than lenght! Cannot split again! len(text) = ' + str(len(text)) + '; lenght = ' + str(
                    lenght))
        half1 = input_list[:len(input_list) / 2]
        half2 = input_list[len(input_list) / 2:]

        output = splitText(half1, output, lenght)
        output = splitText(half2, output, lenght)
    else:
        output.append(input_list)
        return output
    return output


import unittest


class Test(unittest.TestCase):
    def testSplitText(self):
        lenght = 100
        input_list = ["1 2 3 4 5 6 7 8 9 10",
                      "11 12 13 14 15 16 17 18 19 20",
                      "21 22 23 24 25 26 27 28 29 30",
                      "31 32 33 34 35 36 37 38 39 40"]

        expected_result = [['1 2 3 4 5 6 7 8 9 10', '11 12 13 14 15 16 17 18 19 20'], ['21 22 23 24 25 26 27 28 29 30'], ['31 32 33 34 35 36 37 38 39 40']]
        output = []
        result = splitText(input_list, output, lenght=60)
        self.assertEqual(result, expected_result)


if __name__ == '__main__':
    unittest.main()
