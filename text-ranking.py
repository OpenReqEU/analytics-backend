import pandas as pd

from core.utility.serializator import load_obj, save_obj
from core.micro_services import text_ranking_ms
from core import TopicLabeler
from gensim.models.word2vec import Word2Vec


import core.configurations
import core.utility.logger as logger
conf = core.configurations.get_conf()
log_path = conf.get('MAIN', 'log_path')
log_file_name = conf.get('MAIN', 'log_file_name')
log = logger.getLogger(__name__, log_path, log_file_name)

if __name__ == "__main__":
    document_path_file = conf.get('MAIN', 'path_document')
    df = pd.read_csv(document_path_file, encoding='utf-8', error_bad_lines=False)

    df = df[df['idriferimento_ricerca'].isin([5,6])]
    input_list = df['messaggio'].tolist()

    # text_ranking_ms.trainingBigram(input_list, new_model=False)
    bigram_model = load_obj(conf.get('MAIN', 'path_pickle_bigram_model'))

    keywds = text_ranking_ms.extractKeywords(input_list, bigram_model)
    # save_obj(keywds, './bin/keywds.pikle')
    # keywds = load_obj('./bin/keywds.pikle')
    print keywds

    model_w2v = Word2Vec.load(conf.get('MAIN', 'path_pickle_w2v_model'))

    topics = TopicLabeler.textRanking(keywds, model_w2v)
    print topics

