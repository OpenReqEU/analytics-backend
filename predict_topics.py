#!/usr/bin/env python2
"""
HR Text Mining
Claudia Rapuano - Roma
"""

import pandas as pd

from core.utility.serializator import save_obj, load_obj
from core.micro_services import word2vec_ms, clean_text_ms
import core.configurations
import core.utility.logger as logger
import core.topics as Topics

conf = core.configurations.get_conf()
log_path = conf.get('MAIN', 'log_path')
log_file_name = conf.get('MAIN', 'log_file_name')
log = logger.getLogger(__name__, log_path , log_file_name)

def main():

    log.info("--------------------------DRY TOPICS------------------------------------")
    input_list = pd.read_csv(conf.get("MAIN", "path_document"), encoding='utf-8', error_bad_lines=False)
    input_list = input_list[(input_list.idriferimento_ricerca == 5) | (input_list.idriferimento_ricerca == 6)]['messaggio'].tolist()

    w2v_model = word2vec_ms.Word2Vec.load(conf.get('MAIN', 'path_pickle_w2v_model'))
    som_model = load_obj(conf.get('MAIN', 'path_pickle_som_model'))
    cluster_model = load_obj(conf.get('MAIN', 'path_pickle_codebook_cluster_model'))

    dried_topics = Topics.doSomAndDryTopics(input_list, w2v_model, som_model, cluster_model)
    Topics.predictTopics(input_list, w2v_model, som_model, cluster_model, dried_topics)


if __name__ == '__main__':
    main()
