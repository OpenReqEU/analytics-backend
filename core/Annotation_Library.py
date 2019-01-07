# -*- coding: UTF-8 -*-
"""This module contains the functions which make the call to the DBPedia Spotlight
web service annotator and returns the text annotated with entities from DBPedia as a
csv file"""

import re
import urlparse
import os
import requests
from requests.auth import HTTPProxyAuth
import json
from itertools import chain
from gc import collect

# !/usr/bin/env python2
# -*- coding: utf-8 -*-


import requests
import json
import time
import os

baseAdress = "http://193.109.207.65:15024/nlp/"  # DEPLOYED MN01 APP

def annotateSentence(text, language="en", confidence=0.25):
    my_data = {'document': text, 'confidence': confidence}
#    content, _  = tester_app(endpoint='keywords/supervised', baseAdress=baseAdress, reqType='POST', data=my_data, proxies=None)

    req = requests.post(os.path.join(baseAdress, 'keywords/supervised'), json.dumps(my_data), None)

    out_uri = []
    out_entities = []
    out_text = ""

    try:
        resp = req.json()

        if not req.ok:
            print ("ERROR: " + resp.get('error').get('description'))

        content = resp.get('content')

        out_uri = content['dbpediaUri']
        out_entities = content['dbpediaEntities']
        out_text = content['normalizedDocument']
    except:
        pass

    return out_uri, out_entities, out_text, text