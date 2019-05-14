 # -*- coding: utf-8 -*-
"""
Engineering Ingegneria Informatica SPA
Big Data & Analytics Competency Center
"""

# packages
import re

from core.TextCleaner import TextCleaner
from  core import genspacyrank


def extract_text_rank(text):
    '''
    This function extracts unsupervised keywords from text using spacy and networkx pagerank.
    
    Args:
        'text' (string): text of a document
    Returns:
        'keywords' (list): list of keywords
    '''
    text = re.sub('[0-9]', 'x',text)
    text = unicode(text)
    tc = TextCleaner(rm_punct=True, rm_tabs=True, rm_newline=True, rm_digits=False,
                     rm_hashtags=False, rm_tags=True, rm_urls=True, tolower=True, rm_html_tags=True)
    text = tc.regex_applier(text)
    text = text.strip()
    text = ' '.join(text.split())

    keywds, graph = genspacyrank.spacy_text_rank(text, lang='it', rm_stopwords=True, selected_pos=['V', 'N', 'A'],
                                                 topn='all', score=True)
    keywords = []
    for k in keywds:
        keywords.append(k[0])
    
    return keywords
    

def docs_similarity(doc1, doc2, w2v_model, split=False, metrics='cosine'):
    '''
    This function compute the similarity between documents.
    
    Args
        'doc1' (string or list of string): text of a document or list of words of a document
        'doc2' (string or list of string): text of a document or list of words of a document
        'w2v_model' (pickle): word2vec pre-trained model 
        'split' (bool): split text in a list of words  
        'metrics': if wmd compute wmdistance, otherwise the cosine similarity
    Returns:
        float value of similarity between two documents 
    '''
    if split:
        doc1 = doc1.split()
        doc2 = doc2.split()
    doc1 = [word for word in doc1 if word in w2v_model.wv.vocab]
    doc2 = [word for word in doc2 if word in w2v_model.wv.vocab]
    if metrics == 'wmd':
        return w2v_model.wv.wmdistance(document1=doc1, document2=doc2)
    else:
        return w2v_model.wv.n_similarity(ws1=doc1, ws2=doc2)


    
if __name__ == "__main__":  
    
    main()