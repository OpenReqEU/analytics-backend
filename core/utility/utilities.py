import re
import numpy as np
import codecs

from threading import Lock, Thread
lock = Lock()

def convertListOfNumToString(list):
    return ' '.join(str(e) for e in list)

def buildHtml(url, file_name):
    html = None
    if (url != None):
        f = codecs.open(file_name, 'r')
        html = f.read()
    return html

def convertSentenceToListOfWords(sentence):
    # return re.sub("[^\w]", " ", sentence).split()
    return sentence.split()

def meanOfVectors(vectors):
    first_time = True
    check = False
    for v in vectors:
        check = True
        if first_time:
            mean = np.array(v)
            first_time = False
        else:
            mean = mean + np.array(v)
    if (check == True):
        mean = mean / len(vectors)
        return mean.tolist()
    else:
        return []

def concatWords(list_words, separator = " "):
    sentence = separator.join(list_words)
    return sentence

def randomColor():
    import random
    r = lambda: random.randint(0, 255)
    string = ('#%02X%02X%02X' % (r(), r(), r()))
    return string

def convertLisfOfListToList(l):
    flat_list = [item for sublist in l for item in sublist]
    return flat_list

def getUniqueIdentifier():
    import time
    lock.acquire()
    identifier = int(time.time())
    lock.release()
    return identifier

