#!/usr/bin/env python2
"""
HR Text Mining
Claudia Rapuano - Roma
"""

from gensim.models.word2vec import Word2Vec

# custom functions
from core.utility.serializator import save_obj, load_obj
import core.utility.logger as logger
import core.configurations
from core.micro_services import som_ms, word2vec_ms

conf = core.configurations.get_conf()
log_path = conf.get('MAIN', 'log_path')
log_file_name = conf.get('MAIN', 'log_file_name')
log = logger.getLogger(__name__, log_path , log_file_name)

def main():

    log.info("---------------------------CLUSTER CODEBOOK------------------------------------")

    #-------------------------KMEANS --------------------------------------------------
    log.info("START CLUSTERING")
    mySom = load_obj(conf.get('MAIN', 'path_pickle_som_model'))

    som_ms.trainCodebookCluster(mySom, new_model=False)


if __name__ == '__main__':
    main()
