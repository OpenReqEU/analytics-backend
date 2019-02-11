#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 17:18:26 2018

Engineering Ingegneria Informatica spa
Big Data & Analytics Competency Center

@author: francesco pareo
@location: bologna
"""
# packages
import spacy
from collections import namedtuple
import networkx as nx
from gensim.models.phrases import Phrases
from gensim.summarization.summarizer import summarize
import re
import os

import sys
reload( sys)
sys.getdefaultencoding()
from operator import itemgetter

from TextCleaner import TextCleaner

import core.configurations
import core.utility.logger as logger
conf = core.configurations.get_conf()
log_path = conf.get('MAIN', 'log_path')
log_file_name = conf.get('MAIN', 'log_file_name')
log = logger.getLogger(__name__, log_path, log_file_name)


global DEBUG
DEBUG = False


ParsedGraf = namedtuple('ParsedGraf', 'graf_id, graf')
WordNode = namedtuple('WordNode', 'idx, raw, root, pos, dep, keep, word_id ')
RankedLexeme = namedtuple('RankedLexeme', 'text, rank, ids, pos, count')


def spacy_analysis ( text, lang, rm_stopwords=False, selected_pos= ['V', 'N', 'J']):
    """
        tags, pos and lemmas extraction using spacy
    """
    text = unicode(text)
    try:
        spacy_nlp = spacy.load(lang)
    except IOError as e:
        spacy_nlp = spacy.load('en')
        if DEBUG:
            print("Download using bash command: python -m spacy download '"+lang+"'")
            
    doc = spacy_nlp(text)
    
    markup = []
    graf_id = 0
    word_id = 0
    vnj_id = 0
    # sentence iterator
    for span in doc.sents:
        graf = []
        for tag_idx in range(span.start, span.end):
            token = doc[tag_idx]
            
            if type(rm_stopwords)==list and token.text in rm_stopwords:
                continue            
            elif rm_stopwords and token.is_stop:
                continue
            else:
                word = WordNode(idx=word_id, raw=token.text, root=token.lemma_, pos=token.pos_,dep=token.dep_ ,keep=0, word_id=0)

            if token.pos_[0] in selected_pos:
                word = word._replace(word_id=vnj_id, keep=1)
                vnj_id+=1
            

            graf.append(word)
            word_id+=1
            
        markup.append(ParsedGraf(graf_id=graf_id, graf=graf))
        graf_id+=1

    return markup
    
    
def get_tiles (graf, size=3):
    """
    generate word pairs for the TextRank graph
    """
    keeps = list(filter(lambda w: w.keep!=0, graf))
    keeps_len = len(keeps)

    for i in iter(range(0, keeps_len - 1)):
        w0 = keeps[i]

        for j in iter(range(i + 1, min(keeps_len, i + 1 + size))):
            w1 = keeps[j]

            if (w1.idx - w0.idx) <= size:
                yield (w0.root, w1.root)   

    
def spacy_text_rank(text, lang, rm_stopwords=False, selected_pos=['V', 'N', 'J'], topn='all', score=True):
    """
    This function extracts unsupervised keywords from text using spacy and networkx pagerank.
    
    Args:
        'text' (string):
            
        'lang' (string): language package for spacy, please downaload using bash command: 'python -m spacy download 'en' '
                   
        'selected_pos' (list): select only words with indicated Part-of-Speech (only first letter in uppercase) 

        'rm_stopwords' (bool or list): removes stopwords in text
            'True' : remove stopwords using spacy
             list : remove stopwords contained in list
             
        'topn' (int): if topn is not 'all', topn keywords are score-selected
        
        'score' (bool): return keywords score or not
      
    Returns:
       'keywds' (list): tuple list(key,score) containing keywords and page rank score, or olnly list of keys if score=False
       'graph' (networkx obj): page rank network representation of the text
    """
    markup = spacy_analysis ( text, lang, rm_stopwords, selected_pos)
    
    graph = nx.DiGraph()

    for meta in markup:
        for pair in get_tiles(meta.graf):
                
            if not graph.has_node(pair[0]): 
                graph.add_node(pair[0])
            if not graph.has_node(pair[1]): 
                graph.add_node(pair[1])
            
            try:
                graph.edges[pair]["weight"] += 1.0
            except KeyError:
                graph.add_edge(pair[0], pair[1], weight=1.0)
    
    ranks = nx.pagerank(graph)
    
    # sort by score
    ranks = sorted(ranks.items(), key=itemgetter(1),reverse=True)   

    if topn!='all':
        ranks = ranks[0:topn]
    
    if score==False:
        ranks = dict(ranks)
        ranks = ranks.keys()
        
    return ranks, graph




def training_ngram(corpus, lang, train_trigram_model=True, min_count=1, threshold=2, max_vocab_size=40000000, 
                  delimiter='_', progress_per=10000, scoring='default',rm_stopwords=True):
    """
    This function trains a ngram (bigram and trigram) model with gensim phraser. The corpus is tokenized using 
    spacy package, please downaload language package using bash command e.g. python -m spacy download 'en'.
    
    Args:
        'corpus' (list of string): list of texts, path of file
        
        'lang' (string): language package for spacy, please downaload using bash command: 'python -m spacy download 'en' '
        
        'train_trigram_model' (bool): train a trigram model also

        'rm_stopwords' (bool or list): removes stopwords in text
            'True' : remove stopwords using spacy
             list : remove stopwords contained in list
            
        'min_count' (int): ignore all words and bigrams with total collected count lower
        than this.

        'threshold' (int): represents a score threshold for forming the phrases (higher means
        fewer phrases). A phrase of words 'a' followed by 'b' is accepted if the score of the
        phrase is greater than threshold. see the 'scoring' setting.

        'max_vocab_size' (int): is the maximum size of the vocabulary. Used to control
        pruning of less common words, to keep memory under control. The default
        of 40M needs about 3.6GB of RAM; increase/decrease 'max_vocab_size' depending
        on how much available memory you have.

        'delimiter' (string): is the glue character used to join collocation tokens, and
        should be a byte string (e.g. b'_').

        'scoring' (string): specifies how potential phrases are scored for comparison to the 'threshold'
        setting. 'scoring' can be set with either a string that refers to a built-in scoring function,
        or with a function with the expected parameter names. Two built-in scoring functions are available
        by setting 'scoring' to a string:

        'default': from "Efficient Estimaton of Word Representations in Vector Space" by
                   Mikolov, et. al.:
                   (count(worda followed by wordb) - min_count) * N /
                   (count(worda) * count(wordb)) > threshold', where 'N' is the total vocabulary size.
        'npmi': normalized pointwise mutual information, from "Normalized (Pointwise) Mutual
                Information in Colocation Extraction" by Gerlof Bouma:
                ln(prop(worda followed by wordb) / (prop(worda)*prop(wordb))) /
                - ln(prop(worda followed by wordb)
                where prop(n) is the count of n / the count of everything in the entire corpus.

        'npmi' is more robust when dealing with common words that form part of common bigrams, and
        ranges from -1 to 1, but is slower to calculate than the default.

      
    Returns:
        bigram_model, trigram_model (pickle): models in pickle format
    """
    
    # textcleaner instance
    log.info("Cleaning Text...")
    tc = TextCleaner(rm_punct = True, rm_tabs = True, rm_newline = True, rm_digits = False,
                     rm_hashtags = True, rm_tags = True, rm_urls = True, tolower=True, rm_html_tags = True)
    
    spacy_nlp = spacy.load(lang)
    
    def spacycleaner(text):
        text = tc.regex_applier(text)
        text = unicode(text)
        
        # tokenizer and stopwords removal
        doc = spacy_nlp(text)
        token_list = []
        for token in doc:
            if type(rm_stopwords)==list and token.text in rm_stopwords:
                continue
            elif rm_stopwords and token.is_stop:
                continue
            else:
                token_list.append( token.text ) 
        return token_list
    
    
    class text_iterator(object):
        def __init__(self, dirname):
            self.dirname = dirname
    
        def __iter__(self):
            if isinstance(self.dirname,str):
                with open(self.dirname) as reader:
                    for line in reader:
                        yield spacycleaner(line.strip()) 
            else:
               for line in self.dirname:
                   yield spacycleaner(line) 
             

    
    sentences = text_iterator(dirname=corpus)
    log.info("Training Model Phrases...")
    bigram_model = Phrases(sentences,min_count, threshold, max_vocab_size, delimiter, progress_per, scoring)
    if train_trigram_model:
        trigram_model = Phrases(bigram_model[sentences],min_count, threshold, max_vocab_size, delimiter, progress_per, scoring)
    else:
        trigram_model = []
        
    return  bigram_model, trigram_model


def extract_ngram(text,bigram_model=None,trigram_model=None, clean_text=False):
    """
    This function applies models to text and extracts ngrams.
    
    Args:
        'text' (string): 
        'bigram_model','trigram_model'  (pickle): gensim phraser pre-trained models 
      
    Returns:
        'text' (string) : nomalized text 
        'ngrams' (list): list of ngrams
    """
    
    if clean_text:
        # text cleaning: preserve punctuation for sentence splitting in pytextrank
        tc = TextCleaner(rm_punct = True, rm_tabs = True, rm_newline = True, rm_digits = False,
                         rm_hashtags = True, rm_tags = True, rm_urls = True, tolower=True, rm_html_tags = True)
    
        text = tc.regex_applier(text)  
    
    words = text.split()
    
    if  trigram_model is not None:
        words = trigram_model[bigram_model[words]]
    else:
        words = bigram_model[words]
    
    words =  list(set(words))
    # select only ngram from words list
    ngrams = [ word for word in words if (word.count('_') == 1) | (word.count('_') == 2) ]
    text = ' '.join(words)
    
    return text, ngrams


def extract_keywords(text,lang, bigram_model=None,trigram_model=None,selected_pos=['V', 'N', 'J'],rm_stopwords=True, topn='all', score=True):
    """
    This function extracts unsupervised keywords from text using spacy and networkx pagerank 
    and merges words in bignram and tringram using gensim phrases
    
    Args:
        'text' (string):
            
        'lang' (string): language package for spacy, please downaload using bash command: 'python -m spacy download 'en' '
        
        'bigram_model','trigram_model'  (pickle): gensim phraser pre-trained models (use 'training_ngram' function for a better result )
            
        'selected_pos' (list): select only words with indicated Part-of-Speech (only first letter in uppercase) 

        'rm_stopwords' (bool or list): removes stopwords in text
            'True' : remove stopwords using spacy
             list : remove stopwords contained in list
        
        'topn' (int): if topn is not 'all', topn keywords are score-selected
        
        'score' (bool): return keywords score or not
      
    Returns:
       'keywds' (list): tuple list(key,score) containing keywords and page rank score, or olnly list of keys if score=False
       'graph' (networkx obj): page rank network representation of the text
       'text' (string): text nomalized with ngram
       'ngrams' (list): list of ngrams
    """
    
    text = unicode(text)
    
    # text cleaning: preserve punctuation for sentence splitting in pytextrank
    tc = TextCleaner(rm_punct = False, rm_tabs = True, rm_newline = True, rm_digits = False,
                     rm_hashtags = True, rm_tags = True, rm_urls = True, tolower=True, rm_html_tags = True)

    text = tc.regex_applier(text)  
    if DEBUG:
        print('*** CLEANED TEXT ***   '+text)

    ngrams = None
    # compute ngrams with gesnim phraser
    if  trigram_model is not None:
        # remove punctuation
        tc = TextCleaner(rm_punct = True)
        clean_text = tc.regex_applier(text)   
        _, ngrams = extract_ngram(text=clean_text,bigram_model=bigram_model,trigram_model=trigram_model)
        # normalization: substitute ngrams in original text
        for ngram in ngrams:
            ngram_space = ngram.replace('_',' ')
            text = text.replace(ngram_space, ngram)
            
        if DEBUG:
            print('*** NORMALIZED TEXT ***   '+text)        
    
    # keyword extraction with pagerank
    keywds, graph = spacy_text_rank(text, lang, rm_stopwords, selected_pos, topn, score)  

    # keyword extraction with gensim.summarization
    """
    from gensim import summarization
    try:
        keywds_splitted = summarization.keywords(text, split=True)
        keywds = [key.replace(" ", "_") for key in keywds] 
        text = [key.replace(keywds_splitted, keywds) for key in keywds]
    except (IndexError,ZeroDivisionError,ValueError):
        keywds = [],
        graph = nx.DiGraph()
    """

    return keywds, graph, text, ngrams


def load_phraser_models(models_dir,bigram_model_name,trigram_model_name):
    bigram_model = None
    trigram_model = None
    
    # check models dir    
    if not os.path.isdir(models_dir):
        return bigram_model, trigram_model
    # check bigram model
    elif not os.path.exists(os.path.join(models_dir,bigram_model_name)):
        return bigram_model, trigram_model
    else:
        bigram_model = Phrases.load(os.path.join(models_dir,bigram_model_name))
        # check trigram model
        if os.path.exists(os.path.join(models_dir,trigram_model_name)):
            trigram_model = Phrases.load(os.path.join(models_dir,trigram_model_name))
            
    return bigram_model, trigram_model


def save_phraser_models(models_dir,bigram_model=None,trigram_model=None, bigram_model_name='bigramModel',trigram_model_name='trigramModel'):
    res = list()
    if bigram_model is not None:
        bigram_model.save(os.path.join(models_dir,bigram_model_name))
        res.append('Birgam model saved in: '+ os.path.join(models_dir,bigram_model_name))
    if trigram_model is not None:
        trigram_model.save(os.path.join(models_dir,trigram_model_name))
        res.append('Trirgam model saved in: '+ os.path.join(models_dir,trigram_model_name))
    return res


def regex_iterator(text, regex_list): 
    text = unicode(text) 
    out = [] 
    for regex in regex_list: 
        matched =  re.findall(pattern=regex, string=text, flags=re.U) 
        out.append((regex,len(matched)))
    return out



def text_summarization( text, ratio=0.2):
    """
    This function returns a summarized version of the given text using a variation of the TextRank 
    algorithm (see https://arxiv.org/abs/1602.03606).
    
    Args:
        'text' (string): a text
            
        'ratio' (float): should be a number between 0 and 1 that determines 
                         the percentage of the number of sentences of the original
                         text to be chosen for the summary
      
    Returns:
       'summary' (string): a summary of text
    """
    # textcleaner instance
    tc = TextCleaner(rm_punct = False, rm_tabs = True, rm_newline = True, rm_digits = False,
                     rm_hashtags = True, rm_tags = True, rm_urls = True, tolower=True, rm_html_tags = True)
    
    text = tc.regex_applier(text)
    summary = summarize(text, ratio)
    
    return summary



if __name__ == "__main__":
        
    mytext = ["""Compatibility of systems of linear constraints over the set of natural numbers.
        Criteria of compatibility of a system of linear Diophantine equations, strict inequations, 
        and nonstrict inequations are considered. Upper bounds for components of a minimal set of 
        solutions and algorithms of construction of minimal generating sets of solutions for all 
        types of systems are given. These criteria and the corresponding algorithms for constructing 
        a minimal supporting set of solutions can be used in solving all the considered types systems 
        and systems of mixed types. Linear constraints is a math method.""",
        """Criteria of compatibility of a system of linear Diophantine equations, strict inequations,
        and nonstrict inequations are considered. Upper bounds for components of a minimal set of 
        solutions and algorithms of construction of minimal generating sets of solutions for all types 
        of systems are given. These criteria and the corresponding algorithms for constructing a minimal
        supporting set of solutions can be used in solving all the considered types of systems 
        and systems of mixed types.""", """Compatibility of systems of linear constraints over the set of natural numbers.
        Criteria of compatibility of a system of linear Diophantine equations, strict inequations, 
        and nonstrict inequations are considered. Upper bounds for components of a minimal set of 
        solutions and algorithms of construction of minimal generating sets of solutions for all 
        types of systems are given. These criteria and the corresponding algorithms for constructing 
        a minimal supporting set of solutions can be used in solving all the considered types systems 
        and systems of mixed types. Linear constraints is a math method.""",
        """Criteria of compatibility of a system of linear Diophantine equations, strict inequations,
        and nonstrict inequations are considered. Upper bounds for components of a minimal set of 
        solutions and algorithms of construction of minimal generating sets of solutions for all types 
        of systems are given. These criteria and the corresponding algorithms for constructing a minimal
        supporting set of solutions can be used in solving all the considered types of systems 
        and systems of mixed types."""]
    
    lang = 'en'
    # for debugging
    markup = spacy_analysis ( text=mytext, lang=lang, rm_stopwords=False, selected_pos= ['V', 'N', 'J'])
    
    
    
    bigram_model, trigram_model = training_ngram(corpus=mytext, lang=lang, min_count=1, threshold=2, max_vocab_size=40000000,
                                                 delimiter='_', progress_per=10000, scoring='default',rm_stopwords=True)
    
    
    keywds, graph, text, ngrams = extract_keywords(text=mytext,lang='en',
                                           bigram_model=bigram_model,trigram_model=trigram_model,
                                           selected_pos=['V', 'N', 'J'],rm_stopwords=True)
      
    import pylab as plt
    pos = nx.kamada_kawai_layout( G=graph)    
    nx.draw( graph, pos, with_labels=True, arrows=False, alpha=0.8, font_weight='bold',
            node_color='#00bfff', edge_color='#00bfff',font_color ='#00008b' )
    plt.show()

