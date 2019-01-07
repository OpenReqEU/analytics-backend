# This library is a simple implementation of a function to convert textual
# numbers written in English into their integer representations.
#
# This code is open source according to the MIT License as follows.
#
# Copyright (c) 2008 Greg Hewgill
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import re

Small = {
    'zero': 0,
    'uno': 1,
    'due': 2,
    'tre': 3,
    'quatro': 4,
    'cinque': 5,
    'sei': 6,
    'sette': 7,
    'otto': 8,
    'nove': 9,
    'dieci': 10,
    'undici': 11,
    'dodici': 12,
    'tredici': 13,
    'quattordici': 14,
    'quindici': 15,
    'sedici': 16,
    'diciasette': 17,
    'diciotto': 18,
    'diciannove': 19,
    'venti': 20,
    'trenta': 30,
    'quaranta': 40,
    'cinquanta': 50,
    'sessanta': 60,
    'settanta': 70,
    'ottanta': 80,
    'novanta': 90,
    'vent': 20,
    'trent': 30,
    'quarant': 40,
    'cinquant': 50,
    'sessant': 60,
    'settant': 70,
    'ottant': 80,
    'novant': 90
}

Magnitude = {
    'cento':        100,
    'mille':        1000,
    'milione':      1000000,
    'milioni':      1000000,
}

class NumberException(Exception):
    def __init__(self, msg):
        Exception.__init__(self, msg)

def textIsANum(s):

    for w in Small:
        if w in s.lower():
            return True

    for w in Magnitude:
        if w in s.lower():
            return True

    return False



if __name__ == "__main__":
    print textIsANum('seicentoventiquattro')
