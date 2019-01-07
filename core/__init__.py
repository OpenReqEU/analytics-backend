import os

directory = "bin"
if not os.path.exists(directory):
    os.makedirs(directory)

directory = "bin/word2vec"
if not os.path.exists(directory):
    os.makedirs(directory)

directory = "bin/som"
if not os.path.exists(directory):
    os.makedirs(directory)

directory = "data"
if not os.path.exists(directory):
    os.makedirs(directory)

directory = "data/output"
if not os.path.exists(directory):
    os.makedirs(directory)

directory = "data/output/csv"
if not os.path.exists(directory):
    os.makedirs(directory)

directory = "log"
if not os.path.exists(directory):
    os.makedirs(directory)