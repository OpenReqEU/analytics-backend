 # -*- coding: utf-8 -*-
"""
Engineering Ingegneria Informatica SPA
Big Data & Analytics Competency Center
"""

# packages
import re

from core.TextCleaner import TextCleaner
from  core import genspacyrank
import pandas as pd


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

        
def classify_tweet(text,class_definition,model_w2v,model_xgb):
    '''
    This function extracts the keywords from the text, computes the similarity between the text and each class definition, 
    and returns the first two most likely classes, with corresponding score.  
    
    Args
        'text' (string): text of a tweet
        'class_definition' (dict): dictionary of the keywords that define each class
        'model_w2v' (pickle): word2vec pre-trained model
        'model_xgb' (pickle): xgboost pre-trained model
    Returns:
        'predicted_classes' (list): list of the first two most likely classes 
        'scores' (list): list of the scrores of the first two most likely classes 
    '''    
    # keyword extraction
    keywds = extract_text_rank(text)

    # compute distances
    ordered_keys = sorted(list(class_definition.keys()))
    distance = pd.DataFrame([],columns=['class','distance'])
    for _key in ordered_keys:
        cos_sim = docs_similarity(doc1=keywds, doc2=class_definition[_key], w2v_model=model_w2v, split=False,
                                   metrics='cosine')
        new_dis = pd.DataFrame([[_key,cos_sim]],columns=['class','distance'])
        distance = pd.concat([distance,new_dis],ignore_index = True)

    # predict class with xgboost
    distance.index = distance['class']
    df_row = distance.drop('class',axis=1).T
    predictions = model_xgb.predict_proba(df_row.values)
    df_predictions = pd.DataFrame(predictions,columns=['Churn', 'Fatturazione', 'Offerta', 'Problemi', 'Problemi di Rete'])
    
    predicted_classes = []
    scores = []
    for i in [0,1]:
        predicted_classes.append(df_predictions.apply(lambda x: x.sort_values(ascending=False)[:2].index[i],axis=1)[0])
        scores.append(df_predictions.apply(lambda x: x.sort_values(ascending=False)[:2][i],axis=1)[0])
    
    return predicted_classes, scores
    
    
if __name__ == "__main__":  
    
    main()