#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
TextCleaner with regex
Francesco Pareo e Matteo Sartori - Bologna
https://production.eng.it/scm/svnrepos/bdsl/Python_Libraries/ENG_TextCleaning/TextCleaner.py
"""

import re


class TextCleaner(object):

    def __init__(self, rm_punct=True, rm_tabs=True, rm_newline=True, rm_digits=True,
                 tolower=True, rm_hashtags=True, rm_tags=True, rm_urls=True,
                 rm_html_tags=True, rm_elongated_vowels=True, rm_elongated_consonants=True,
                 rm_words_short_words=True):
        self.rm_punct = rm_punct
        self.rm_tabs = rm_tabs
        self.rm_newline = rm_newline
        self.rm_digits = rm_digits
        self.rm_html_tags = rm_html_tags
        self.tolower = tolower
        self.rm_hashtags = rm_hashtags
        self.rm_tags = rm_tags
        self.rm_urls = rm_urls
        self.rm_elongated_vowels = rm_elongated_vowels
        self.rm_elongated_consonants = rm_elongated_consonants
        self.pattern_list = list()
        self.repl_list = list()
        self.rm_words_short_words = rm_words_short_words

        self.regex_compiler(self)

    @staticmethod
    def regex_compiler(self):

        if self.rm_urls:
            self.pattern_list.append(re.compile(r'http\S+'))
            self.repl_list.append(' ')
            self.pattern_list.append(re.compile(r'www\S+'))
            self.repl_list.append(' ')
        if self.rm_html_tags:
            self.pattern_list.append(re.compile(r'(&#?[A-z0-9]{1,8};)'))
            self.repl_list.append(' ')
        if self.rm_hashtags:
            self.pattern_list.append(re.compile(r'#+[A-z0-9\_]+'))
            self.repl_list.append(' ')
        if self.rm_tags:
            self.pattern_list.append(re.compile(r'@+[A-z0-9\_]+((:)?)'))
            self.repl_list.append(' ')
        if self.rm_tabs:
            self.pattern_list.append(re.compile(r'\t'))
            self.repl_list.append(' ')
            self.pattern_list.append(re.compile(r'\v'))
            self.repl_list.append(' ')
        if self.rm_newline:
            self.pattern_list.append(re.compile(r'\n'))
            self.repl_list.append(' ')
        if self.rm_punct:
            self.pattern_list.append(
                re.compile(ur'[\_[\]\\]|[^0-9A-z\u00E0\u00E1\u00E8\u00E9\u00EC\u00ED\s\u00F2\u00F3\u00F9\u00FA]'))
            self.repl_list.append(' ')
        else:
            self.pattern_list.append(re.compile(
                ur'[\_[\]\\]|[^0-9A-z\u00E0\u00E1\u00E8\u00E9\u00EC\u00ED\s\u00F2\u00F3\u00F9\u00FA!;?.,:\']'))
            self.repl_list.append(' ')
            self.pattern_list.append(re.compile(r"([!?,.:;'])([!?,.:;']+)"))
            self.repl_list.append(r'\1')
        if self.rm_digits:
            self.pattern_list.append(re.compile(r'[0-9]'))
            self.repl_list.append(' ')
        if self.rm_elongated_vowels:
            self.pattern_list.append(re.compile(r"([AEIOUaeiou])\1{2,}"))
            self.repl_list.append(r'\1')
        if self.rm_elongated_consonants:
            self.pattern_list.append(re.compile(r"([qwrtypsdfghjklzxcvbnmQWRTYPSDFGHJKLZXCVBNM])\1{2,}"))
            self.repl_list.append(r'\1\1')


    def regex_applier(self, text):
        for i in range(0, len(self.pattern_list)):
            text = self.pattern_list[i].sub(string=text, repl=self.repl_list[i])
        if self.tolower:
            text = text.lower()
        if self.rm_words_short_words:
            text = re.sub(r'\b\w{1,3}\b', '', text)
        return text

if __name__ == '__main__':
    text = "posso suy cia ioooooo"

    # text cleaning: preserve punctuation for sentence splitting in pytextrank
    tc = TextCleaner(rm_punct = False, rm_tabs = True, rm_newline = True, rm_digits = False,
                     rm_hashtags = True, rm_tags = True, rm_urls = True, tolower=True, rm_html_tags = True)

    print('*** ORIGINAL TEXT ***   '+text)
    text = tc.regex_applier(text)
    print('*** CLEANED TEXT ***   '+text)
