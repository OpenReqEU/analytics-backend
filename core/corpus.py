import core.utility.logger as logger
import core.utility.utilities as utilities
import core.configurations
import os
from core.micro_services import clean_text_ms
from core.utility.serializator import save_obj, load_obj
import core.Annotation_Library
import core.WikipediaExtractorLibrary

conf = core.configurations.get_conf()
log_path = conf.get('MAIN', 'log_path')
log_file_name = conf.get('MAIN', 'log_file_name')
log = logger.getLogger(__name__, log_path, log_file_name)


def createCorpus(cleaned_input_list):
    # -------------------------GET ENTITIES----------------------------------------------------
    log.info("GET ENTITIES")
    entity_list = []
    confidence = conf.getfloat('ENTITY', 'confidence')
    entity_list, tweet_with_entity_list, all_uri = getEntities(cleaned_input_list, confidence=confidence)

    # -------------------------GET WIKIPEDIA PAGES---------------------------------------------
    log.info("GET WIKIPEDIA PAGES")
    wikipage_list = getWikipediaPages(all_uri)
    wikipage_list = clean_text_ms.cleanText(wikipage_list)

    # -------------------------CREATE CORPUS---------------------------------------------------
    print log.info("CREATE CORPUS")
    tweet_corpus = createTweetCorpus(wikipage_list, cleaned_input_list, tweet_with_entity_list)
    corpus = tweet_corpus
    corpus += wikipage_list
    return corpus

# get all entities
def getEntities(cleaned_list, confidence=0.25):
    # list of list of uri, list of list of entities, list of sentence with entity, list of sentence
    out_uri_list, out_entities_list, out_text_list = applyAnnotateSenteceOnList(cleaned_list, confidence=confidence)

    # flat list of list of entity
    all_entity = [val for sublist in out_entities_list for val in sublist]
    # flat list of list of uri
    all_uri = [val for sublist in out_uri_list for val in sublist]

    # removing duplicates in lists
    all_entity = list(set(all_entity))
    all_uri = list(set(all_uri))

    return all_entity, out_text_list, all_uri

def getWikipediaPages(all_uri):
    lang = conf.get('ENTITY', 'lang')
    # get wikipedia text extraction
    #wikipage_list = list(map(lambda (uri): WikipediaExtractorLibrary.extract_Wikipage_from_url(uri, lang), all_uri))
    wikipage_list = []
    count = 0
    for uri in all_uri:
        log.info(str(count)+"/"+str(len(all_uri)))
        wikipage_list.append(core.WikipediaExtractorLibrary.extract_Wikipage_from_url(uri, lang))
        count = count +1
    wikipage_list = [x for x in wikipage_list if x != []]

    return wikipage_list

# create corpus from tweets (enlarge it if needed)
# use tweet with entity if USE_ENTITY_FOR_W2V is True
# tweet and wikipage will have less or more the same size (by repeating tweets)
def createTweetCorpus(wikipage_list, cleaned_tweet_list, tweet_with_entity_list):

    # get number of words of wikipedia articles
    nr_wiki_words = getNumberOfWords(wikipage_list)

    # ------------------- get all tweets ---------------------------------------
    merged_tweets = cleaned_tweet_list

    if (conf.get('FLAG', 'USE_ENTITY_FOR_W2V')):# configurable
        # merge tweets without entity with tweets with entity
        merged_tweets = tweet_with_entity_list + cleaned_tweet_list

    all_tweets = []

    # emulate do while
    while True:
        all_tweets += merged_tweets
        # convert unicode to string
        #all_tweets_words = list(map(lambda (tweet): tweet.encode('UTF8'), all_tweets))
        all_tweets_words = []
        for tweet in all_tweets:
            all_tweets_words.append(str(tweet).encode('UTF8'))
        # get all words of tweets
        all_tweets_words = ' '.join(all_tweets_words)
        all_tweets_words = all_tweets_words.split(' ')
        # get number of words of tweets
        nr_tweet_words = len(all_tweets_words)

        if nr_tweet_words > nr_wiki_words:
            break

    return all_tweets
    # --------------------------------------------------------------------------

# apply method annotateSentence to a list (to get entity)
def applyAnnotateSenteceOnList(list, language="it", confidence=0.2):

    out_uri_list = []
    out_entities_list = []
    out_text_list = []

    for text in list:
        out_uri, out_entities, out_text, sentence = core.Annotation_Library.annotateSentence(text, language=language, confidence=confidence)
        out_uri_list.append(out_uri)
        out_entities_list.append(out_entities)
        out_text_list.append(out_text)

    return out_uri_list, out_entities_list, out_text_list

def getNumberOfWords(sentences):
    # convert list of unicode to list of string
    #sentences = list(map(lambda (page): page.encode('UTF8'), sentences))
    list_sentences = []
    for page in sentences:
        list_sentences.append(page.encode('UTF8'))
    sentences = list_sentences

    # get all words
    words = ' '.join(sentences)
    words = words.split(' ')

    # get number of words
    nr_words = len(words)

    return nr_words