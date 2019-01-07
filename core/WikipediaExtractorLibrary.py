"""
Francesco Pareo - Bologna
"""

import wikipedia
import requests
from bs4 import BeautifulSoup
import re


def extract_Wikipage_from_title(wiki_title, verbose=True):
    try:
        wiki_text = wikipedia.WikipediaPage(wiki_title).content
        return wiki_text
    except:
        return []
        if verbose:
            print(wiki_title, 'not found.')


def extract_Wikipage_from_url(wiki_url, lang, verbose=True):
    wikipedia.set_lang(lang)
    req = requests.get(wiki_url)
    # soup = BeautifulSoup(req.text, "lxml")
    soup = BeautifulSoup(req.text)
    wiki_title = soup.title.string

    #get the part before ' - '
    wiki_title = wiki_title.split(u' \u2014 ')[0]
    #wiki_title = re.sub(string=wiki_title, pattern=' - Wikipedia', repl='')

    return extract_Wikipage_from_title(wiki_title, verbose)
