import matplotlib

matplotlib.use('Agg')

from flask import Flask, request, jsonify
from flasgger import Swagger

import thread

Swagger.DEFAULT_CONFIG = {
    "headers": [
    ],
    "specs": [
        {
            "endpoint": 'apispec_1',
            "route": '/analytics-backend/apispec_1.json',
            "rule_filter": lambda rule: True,  # all in
            "model_filter": lambda tag: True,  # all in
        }
    ],
    "static_url_path": "/analytics-backend/flasgger_static",
    # "static_folder": "static",  # must be set by user
    "swagger_ui": True,
    "specs_route": "/analytics-backend/apidocs/"
}


import os
import socket
import json
import pandas as pd
from flask import send_file
from flask_cors import CORS
from flask import render_template

import core.utility.utilities as utilities
import core.configurations
import core.utility.logger as logger
from core import TopicLabeler, TextCleaner, genspacyrank

conf = core.configurations.get_conf()

from core.micro_services import clean_text_ms, som_ms, word2vec_ms, text_ranking_ms
import core.topics as Topics
import core.corpus as Corpus

from core.docs_similarity import extract_text_rank, docs_similarity
from gensim.models import Word2Vec

# load class definition for tweet classification
file_definition = conf.get('MAIN', 'class_definition')
class_definition = json.load(open(file_definition))
# load word2vec for tweet classification
filename_w2v = conf.get('MAIN', 'path_word2vec_model_classification')
model_w2v = Word2Vec.load(filename_w2v)

app = Flask(__name__)
swagger = Swagger(app)
cors = CORS(app)

log_path = conf.get('MAIN', 'log_path')
log_file_name = conf.get('MAIN', 'log_file_name')
log = logger.getLogger(__name__, log_path, log_file_name)

@app.errorhandler(500)
def internalServerError(error):
    response = jsonify({"error": "Internal server error"})
    response.status_code = 500
    return response


def returnModelStatus(filename, model_id):
    import os.path
    if (os.path.isfile(filename)):
        response = jsonify({"warning": "Model "+str(model_id)+" is still training"})
    else:
        response = jsonify({"warning": "Model "+str(model_id)+" is not trained"})
    response.status_code = 299
    return response

# cleaning
@app.route("/analytics-backend/cleanText", methods=['POST'])
def cleanText():
    """
        Clean text
        Get a list of tweet message, return a list of cleaned messages
        ---
        parameters:
          - in: body
            name: body
            schema:
              type: array
              items:
                type: object
                properties:
                  message:
                    type: string
                    description: tweeet message
            required: true
        responses:
          200:
            description: Text cleaned
            schema:
                type: array
                items:
                    type: string
          500:
            description: Internal Server Error
            schema:
                type: object
                properties:
                    error:
                     type: string
    """
    log.info("/analytics-backend/cleanText")
    r = request
    data_json = json.dumps(request.get_json(silent=True))
    input_list = pd.read_json(data_json, encoding='utf8')['message'].tolist()

    cleaned_tweet_list = clean_text_ms.cleanText(pd.read_json(data_json, encoding='utf8')['message'].tolist())
    return jsonify(cleaned_tweet_list)


# w2v
@app.route("/analytics-backend/getEmbeddedWords", methods=['POST'])
def getEmbeddedWords():
    """
        Word embedding
        Get a the id of word2vec model the list of tweet messages and return a list of vector
        ---
        parameters:
          - in: body
            name: body
            schema:
              type: object
              properties:
                w2v_model_id:
                  type: string
                  description: id of model to use for word embedding
                tweets:
                  type: array
                  items:
                    type: object
                    properties:
                      message:
                        type: string
                        description: tweeet message
            required: true
        responses:
          200:
            description: list of embedded words
            schema:
              type: array
              items:
                type: array
                items:
                  type: string
          500:
            description: Internal Server Error
            schema:
                type: object
                properties:
                    error:
                     type: string
          299:
            description: Model is still training or not trained
            schema:
                type: object
                properties:
                    warning:
                     type: string
    """
    log.info("/analytics-backend/getEmbeddedWords")
    data_json = json.dumps(request.get_json(silent=True))
    data_json = json.loads(data_json)

    model_id = data_json["w2v_model_id"]
    input_list = json.dumps(data_json["tweets"])
    input_list = pd.read_json(input_list, encoding='utf8')['message'].tolist()

    filename = conf.get('MAIN', 'path_pickle_w2v_model_incr_fold') + "word2vec_" + str(model_id) + ".pickle"
    try:
        model = word2vec_ms.Word2Vec.load(filename)
    except:
        filename = conf.get('MAIN', 'path_pickle_w2v_model_incr_fold') + "word2vec_" + str(model_id) + "_training.txt"
        return returnModelStatus(filename, model_id)

    embedded_words_tweets, dict_index2word_tweet, dict_word2index_tweet = word2vec_ms.getEmbeddedWords(
        input_list, model)

    list = embedded_words_tweets.tolist()
    return jsonify(list)


# som - entities
@app.route("/analytics-backend/doSomAndPlot", methods=['POST'])
def doSomAndPlot1():
    """
        Get entities: Apply SOM and plot result of codebook MST
        Get word2vec model id, som model id, the list of tweet messages or the url of the csv with messages, the type of result and return a result graph
        ---
        parameters:
          - in: body
            name: body
            schema:
              type: object
              properties:
                w2v_model_id:
                  type: string
                  description: id of model to use for word embedding
                som_model_id:
                  type: string
                  description: id of SOM model
                type_chart:
                  type: string
                  description: type of result "d3" (html) of json
                url_input:
                  type: string
                  description: url of the csv with messages
                tweets:
                  type: array
                  items:
                    type: object
                    properties:
                      message:
                        type: string
                        description: tweeet message
            required: true
        responses:
          200:
            description: graph of entities
            schema:
              type: object
              properties:
                directed:
                  type: boolean
                graph:
                  type: object
                links:
                  type: array
                  items:
                    type: object
                    properties:
                      source:
                        type: integer
                      target:
                        type: integer
                multigraph:
                  type: boolean
                nodes:
                  type: array
                  items:
                    type: object
                    properties:
                      id:
                        type: integer
                      name:
                        type: string
                      pos:
                        type: array
                        items:
                          type: integer
          500:
            description: Internal Server Error
            schema:
                type: object
                properties:
                    error:
                     type: string
          299:
            description: Model is still training or not trained
            schema:
                type: object
                properties:
                    warning:
                     type: string
    """
    log.info("/analytics-backend/doSomAndPlot")

    # reading json input
    data_json = json.dumps(request.get_json(silent=True))
    data_json = json.loads(data_json)
    type_chart = data_json["type_chart"]
    w2v_model_id = data_json["w2v_model_id"]
    som_model_id = data_json["som_model_id"]

    if 'url_input' in data_json:
        url_input = data_json["url_input"]
        df = pd.read_csv(url_input)

        # DEBUGGING
        debugging = conf.get('MAIN', 'debugging')
        if (debugging == 'True'):
            document_path_file = conf.get('MAIN', 'path_document')
            df = pd.read_csv(document_path_file, encoding='utf-8', error_bad_lines=False)
            df = df.head()

        input_list = df['message'].tolist()
    else:
        input_list = json.dumps(data_json["tweets"])
        input_list = pd.read_json(input_list, encoding='utf8')['message'].tolist()

    filename = conf.get('MAIN', 'path_pickle_w2v_model_incr_fold') + "word2vec_" + str(w2v_model_id) + ".pickle"
    try:
        model = word2vec_ms.Word2Vec.load(filename)
    except:
        filename = conf.get('MAIN', 'path_pickle_w2v_model_incr_fold') + "word2vec_" + str(w2v_model_id) + "_training.txt"
        return returnModelStatus(filename, w2v_model_id)


    # get embedded words from input
    cleaned_tweet_list = clean_text_ms.cleanText(input_list)
    embedded_words, dict_index2word, dict_word2index = word2vec_ms.getEmbeddedWords(
        cleaned_tweet_list, model)

    filename = conf.get('MAIN', 'path_pickle_som_model_incr_fold') + "som_" + str(som_model_id) + ".pickle"
    try:
        som_model = som_ms.load_obj(filename)
    except:
        filename = conf.get('MAIN', 'path_pickle_som_model_incr_fold') + "som_" + str(som_model_id) + "_training.txt"
        return returnModelStatus(filename, som_model_id)

    file_name = conf.get('MAIN', 'MST_html_d3_output_file')
    response = som_ms.doSomAndPlot(som_model, embedded_words, dict_index2word, file_name, type_chart)

    if (type_chart == "d3"):
        return render_template('MST_d3.html')
        return html
    elif (type_chart == "json"):
        return jsonify(response)
    else:
        return internalServerError(500)


# som - topics
@app.route("/analytics-backend/computeTopics", methods=['POST'])
def computeTopics():
    """
        Extracts topics from a list of tweets
        Get word2vec model id, som model id, the codebook cluster model id, the list of tweet messages or the url of the csv with messages and return a result graph
        ---
        parameters:
          - in: body
            name: body
            schema:
              type: object
              properties:
                w2v_model_id:
                  type: string
                  description: id of model to use for word embedding
                som_model_id:
                  type: string
                  description: id of SOM model
                codebook_cluster_model_id:
                  type: string
                  description: id of codebook cluster model
                tweets:
                  type: array
                  items:
                    type: object
                    properties:
                      message:
                        type: string
                        description: tweeet message
            required: true
        responses:
          200:
            description: graphs of topics
            schema:
              type: array
              items:
                type: object
                properties:
                  directed:
                    type: boolean
                  graph:
                    type: object
                  links:
                    type: array
                    items:
                      type: object
                      properties:
                        source:
                          type: integer
                        target:
                          type: integer
                  multigraph:
                    type: boolean
                  nodes:
                    type: array
                    items:
                      type: object
                      properties:
                        id:
                          type: integer
                        name:
                          type: string
                        pos:
                          type: array
                          items:
                            type: integer
          500:
            description: Internal Server Error
            schema:
                type: object
                properties:
                    error:
                     type: string
          299:
            description: Model is still training or not trained
            schema:
                type: object
                properties:
                    warning:
                     type: string
    """
    log.info("/analytics-backend/computeTopics")

    # remove old html topics
    import glob
    import os
    for fl in glob.glob("./templates/dried_*.html"):
        os.remove(fl)

    # reading json input
    data_json = json.dumps(request.get_json(silent=True))
    data_json = json.loads(data_json)

    w2v_model_id = data_json["w2v_model_id"]
    som_model_id = data_json["som_model_id"]
    cluster_model_id = data_json["codebook_cluster_model_id"]
    input_list = json.dumps(data_json["tweets"])
    input_list = pd.read_json(input_list, encoding='utf8')['message'].tolist()

    # load models
    filename = conf.get('MAIN', 'path_pickle_w2v_model_incr_fold') + "word2vec_" + str(w2v_model_id) + ".pickle"
    try:
        w2v_model = word2vec_ms.Word2Vec.load(filename)
    except:
        filename = conf.get('MAIN', 'path_pickle_w2v_model_incr_fold') + "word2vec_" + str(w2v_model_id) + "_training.txt"
        return returnModelStatus(filename, w2v_model_id)

    filename = conf.get('MAIN', 'path_pickle_som_model_incr_fold') + "som_" + str(som_model_id) + ".pickle"
    try:
        som_model = som_ms.load_obj(filename)
    except:
        filename = conf.get('MAIN', 'path_pickle_som_model_incr_fold') + "som_" + str(som_model_id) + "_training.txt"
        return returnModelStatus(filename, som_model_id)

    filename = conf.get('MAIN', 'path_pickle_codebook_cluster_model_incr_fold') + "codebook_cluster_" + str(
        cluster_model_id) + ".pickle"
    try:
        cluster_model = som_ms.load_obj(filename)
    except:
        conf.get('MAIN', 'path_pickle_codebook_cluster_model_incr_fold') + "codebook_cluster_" + str(
            cluster_model_id) + "_training.txt"
        return returnModelStatus(filename, cluster_model_id)

    dried_topics = Topics.doSomAndDryTopics(input_list, w2v_model, som_model, cluster_model)
    graphs = Topics.predictTopics(input_list, w2v_model, som_model, cluster_model, dried_topics, type_chart="json")

    response = jsonify(graphs)
    return response


# plot
@app.route("/analytics-backend/getCodebookActivation")
def getCodebookActivation():
    """
            Plot result of codebook Activation
            Get som model id and return a png of codebook activations
            ---
            consumes:
              - application/json
            produces:
              - application/json
            parameters:
              - in: query
                name: som_model_id
                type: string
                required: true
                description: id of SOM model
            responses:
              200:
                description: png of codebook activations
                schema:
                    type: object
                    properties:
                        pictures:
                         type: string
              500:
                description: Internal Server Error
                schema:
                    type: object
                    properties:
                        error:
                         type: string
              299:
                description: Model is still training or not trained
                schema:
                    type: object
                    properties:
                        warning:
                         type: string
    """
    log.info("/analytics-backend/getCodebookActivation")
    som_model_id = request.args.get('som_model_id')

    filename = conf.get('MAIN', 'path_pickle_som_model_incr_fold') + "som_" + str(som_model_id) + ".pickle"
    try:
        som_model = som_ms.load_obj(filename)
    except:
        filename = conf.get('MAIN', 'path_pickle_som_model_incr_fold') + "som_" + str(som_model_id) + "_training.txt"
        return returnModelStatus(filename, som_model_id)

    png = som_ms.getCodebookActivation(som_model)

    # f = open(png, 'r+')
    # data = f.read()

    import base64
    with open(png, "rb") as imageFile:
        s = base64.b64encode(imageFile.read())

    return json.dumps({'picture': s})
    # return send_file(png, mimetype='image/gif')


# plot
@app.route("/analytics-backend/getCellFrequencyDistribution", methods=['POST'])
def getCellFrequencyDistribution():
    """
        Plot frequency distribution
        Get word2vec model id, som model id, the list of tweet messages or the url of the csv with messages, the type of result ("bubble" or "bar") and return a result graph
        ---
        consumes:
          - application/json
        produces:
          - application/json
        parameters:
          - in: body
            name: body
            schema:
              type: object
              properties:
                w2v_model_id:
                  type: string
                  description: id of model to use for word embedding
                som_model_id:
                  type: string
                  description: id of SOM model
                num:
                  type: string
                  description: number of variables sorted from biggest
                type_chart:
                  type: string
                  description: type of result "bubble" or "bar"
                url_input:
                  type: string
                  description: url of the csv with messages
                tweets:
                  type: array
                  items:
                    type: object
                    properties:
                      message:
                        type: string
                        description: tweeet message
            required: true
        responses:
          200:
            description: graph of frequencies
            schema:
                type: object
                properties:
                    html:
                     type: string
          500:
            description: Internal Server Error
            schema:
                type: object
                properties:
                    error:
                     type: string
          299:
            description: Model is still training or not trained
            schema:
                type: object
                properties:
                    warning:
                     type: string
    """
    log.info("/analytics-backend/getCellFrequencyDistribution")
    data_json = json.dumps(request.get_json(silent=True))
    data_json = json.loads(data_json)

    type_chart = data_json["type_chart"]
    num = data_json["num"]
    w2v_model_id = data_json["w2v_model_id"]
    som_model_id = data_json["som_model_id"]

    if 'url_input' in data_json:
        url_input = data_json["url_input"]
        df = pd.read_csv(url_input)

        # DEBUGGING
        debugging = conf.get('MAIN', 'debugging')
        if (debugging == 'True'):
            document_path_file = conf.get('MAIN', 'path_document')
            df = pd.read_csv(document_path_file, encoding='utf-8', error_bad_lines=False)
            df = df.head()

        input_list = df['message'].tolist()
    else:
        input_list = json.dumps(data_json["tweets"])
        input_list = pd.read_json(input_list, encoding='utf8')['message'].tolist()

    filename = conf.get('MAIN', 'path_pickle_w2v_model_incr_fold') + "word2vec_" + str(w2v_model_id) + ".pickle"
    try:
        w2v_model = word2vec_ms.Word2Vec.load(filename)
    except:
        filename = conf.get('MAIN', 'path_pickle_w2v_model_incr_fold') + "word2vec_" + str(w2v_model_id) + "_training.txt"
        return returnModelStatus(filename, w2v_model_id)

    filename = conf.get('MAIN', 'path_pickle_som_model_incr_fold') + "som_" + str(som_model_id) + ".pickle"
    try:
        som_model = som_ms.load_obj(filename)
    except:
        filename = conf.get('MAIN', 'path_pickle_som_model_incr_fold') + "som_" + str(som_model_id) + "_training.txt"
        return returnModelStatus(filename, som_model_id)

    html = som_ms.getCellFrequencyDistribution(input_list, w2v_model, som_model, num, type_chart)
    return json.dumps({'html': html})
    # return html


# plot
@app.route("/analytics-backend/getUmatrix")
def getUmatrix():
    """
            Plot result of umatrix
            Get som model id and return a png of Umatrix
            ---
            consumes:
              - application/json
            produces:
              - application/json
            parameters:
              - in: query
                name: som_model_id
                type: string
                required: true
                description: id of SOM model
            responses:
              200:
                description: png of Umatrix
                schema:
                    type: object
                    properties:
                        picture:
                         type: string
              500:
                description: Internal Server Error
                schema:
                    type: object
                    properties:
                        error:
                         type: string
              299:
                description: Model is still training or not trained
                schema:
                    type: object
                    properties:
                        warning:
                         type: string
    """
    log.info("/analytics-backend/getUmatrix")
    som_model_id = request.args.get('som_model_id')

    filename = conf.get('MAIN', 'path_pickle_som_model_incr_fold') + "som_" + str(som_model_id) + ".pickle"
    try:
        som_model = som_ms.load_obj(filename)
    except:
        filename = conf.get('MAIN', 'path_pickle_som_model_incr_fold') + "som_" + str(som_model_id) + "_training.txt"
        return returnModelStatus(filename, som_model_id)

    png = som_ms.getUmatrix(som_model)

    import base64
    with open(png, "rb") as imageFile:
        s = base64.b64encode(imageFile.read())

    return json.dumps({'picture': s})
    # return send_file(png, mimetype='image/gif')


# plot
@app.route("/analytics-backend/getCostOfSom", methods=['GET'])
def getCostOfSom():
    """
            Get cost of Som
            Get som model id and return cost of model
            ---
            parameters:
              - in: query
                name: som_model_id
                type: string
                required: true
                description: id of SOM model
            responses:
              200:
                description: cost of som
                schema:
                    type: object
                    properties:
                      cost of model:
                        type: string
              500:
                description: Internal Server Error
                schema:
                    type: object
                    properties:
                        error:
                         type: string
              299:
                description: Model is still training or not trained
                schema:
                    type: object
                    properties:
                        warning:
                         type: string
    """
    log.info("/analytics-backend/getCostOfSom")
    som_model_id = request.args.get('som_model_id')

    filename = conf.get('MAIN', 'path_pickle_som_model_incr_fold') + "som_" + str(som_model_id) + ".pickle"
    try:
        som_model = som_ms.load_obj(filename)
    except:
        filename = conf.get('MAIN', 'path_pickle_som_model_incr_fold') + "som_" + str(som_model_id) + "_training.txt"
        return returnModelStatus(filename, som_model_id)


    cost = som_ms.getCostOfSom(som_model)
    response = jsonify({"cost of model": cost})
    return response


@app.route("/analytics-backend/getCodebookWords", methods=['POST'])
def getCodebookWords():
    """
        Get all words associated to codebooks
        Get a the id of word2vec model, the id of SOM model and the list of tweet messages and return the lists of associated words
        ---
        parameters:
          - in: body
            name: body
            schema:
              type: object
              properties:
                w2v_model_id:
                  type: string
                  description: id of model to use for word embedding
                som_model_id:
                  type: string
                  description: id of SOM model
                tweets:
                  type: array
                  items:
                    type: object
                    properties:
                      message:
                        type: string
                        description: tweeet message
            required: true
        responses:
          200:
            description: list of codebook words
            schema:
              type: object
              properties:
                0:
                  type: array
                  items:
                    type: string
                  description: codebook word
                1:
                  type: array
                  items:
                    type: string
                  description: codebook word
                ...:
                  type: array
                  items:
                    type: string
                  description: codebook word
                n:
                  type: array
                  items:
                    type: string
                  description: codebook word
          500:
            description: Internal Server Error
            schema:
                type: object
                properties:
                    error:
                     type: string
          299:
            description: Model is still training or not trained
            schema:
                type: object
                properties:
                    warning:
                     type: string
    """
    log.info("/analytics-backend/getCodebookWords")
    # reading json input
    data_json = json.dumps(request.get_json(silent=True))
    data_json = json.loads(data_json)

    w2v_model_id = data_json["w2v_model_id"]
    som_model_id = data_json["som_model_id"]
    input_list = json.dumps(data_json["tweets"])
    input_list = pd.read_json(input_list, encoding='utf8')['message'].tolist()

    filename = conf.get('MAIN', 'path_pickle_w2v_model_incr_fold') + "word2vec_" + str(w2v_model_id) + ".pickle"
    try:
        w2v_model = word2vec_ms.Word2Vec.load(filename)
    except:
        filename = conf.get('MAIN', 'path_pickle_w2v_model_incr_fold') + "word2vec_" + str(
            w2v_model_id) + "_training.txt"
        return returnModelStatus(filename, w2v_model_id)

    filename = conf.get('MAIN', 'path_pickle_som_model_incr_fold') + "som_" + str(som_model_id) + ".pickle"
    try:
        som_model = som_ms.load_obj(filename)
    except:
        filename = conf.get('MAIN', 'path_pickle_som_model_incr_fold') + "som_" + str(som_model_id) + "_training.txt"
        return returnModelStatus(filename, som_model_id)

    codebook2words = som_ms.getCodebookWords(input_list, w2v_model, som_model)
    response = jsonify(codebook2words)
    return response

@app.route("/analytics-backend/dried")
def showTopic():
    # """
    #     Show the graph associated to the topic
    #     Get a the id of the topic to show and return the html of the graph
    #     ---
    #     produces:
    #       - text/html
    #     parameters:
    #       - in: query
    #         name: num_graph
    #         type: string
    #         required: true
    #     responses:
    #       200:
    #         description: graph of topic
    #       500:
    #         description: Internal Server Error
    #         schema:
    #             type: object
    #             properties:
    #                 error:
    #                  type: string
    # """
    # """
    #     Get MST of topic
    #     ---
    #     parameters:
    #         - num_graph: num
    #     responses:
    #       200:
    #         description: MST of topic number num
    #         content: text/json
    # """
    num_graph = request.args.get('num_graph')
    return render_template('dried_' + num_graph + '.html')
    return html

@app.route("/analytics-backend/keywordsExtraction", methods=['POST'])
def keywordsExtraction():
    """
        Keywords Extraction
        Get a the id of the bigram model, the list of tweet messages or the url of the csv with tweet messages and return the lists of keywords
        ---
        parameters:
          - in: body
            name: body
            schema:
              type: object
              properties:
                url_input:
                  type: string
                  description: url of csv with tweet messages
                bigram_model_id:
                  type: string
                  description: id of bigram model
                tweets:
                  type: array
                  items:
                    type: object
                    properties:
                      message:
                        type: string
                        description: tweeet message
            required: true
        responses:
          200:
            description: lists of keywords
            schema:
              type: array
              items:
                type: array
                items:
                  type: string
          500:
            description: Internal Server Error
            schema:
                type: object
                properties:
                    error:
                     type: string
          299:
            description: Model is still training or not trained
            schema:
                type: object
                properties:
                    warning:
                     type: string
    """
    log.info("/analytics-backend/keywordsExtraction")
    data_json = json.dumps(request.get_json(silent=True))
    data_json = json.loads(data_json)

    if 'url_input' in data_json:
        url_input = data_json["url_input"]
        df = pd.read_csv(url_input)

        # DEBUGGING
        debugging = conf.get('MAIN', 'debugging')
        if (debugging == 'True'):
            document_path_file = conf.get('MAIN', 'path_document')
            df = pd.read_csv(document_path_file, encoding='utf-8', error_bad_lines=False)
            df = df.head()

        input_list = df['message'].tolist()
    else:
        input_list = json.dumps(data_json["tweets"])
        input_list = pd.read_json(input_list, encoding='utf8')['message'].tolist()


    bigram_model_id = data_json["bigram_model_id"]

    filename = conf.get('MAIN', 'path_pickle_bigram_model_incr_fold') + "bigram_" + str(bigram_model_id) + ".pickle"
    try:
        bigram_model = som_ms.load_obj(filename)
    except:
        filename = conf.get('MAIN', 'path_pickle_bigram_model_incr_fold') + "bigram_" + str(
            bigram_model_id) + "_training.txt"
        return returnModelStatus(filename, bigram_model_id)

    keywds = text_ranking_ms.extractKeywords(input_list, bigram_model)

    response = jsonify(keywds)
    return response

@app.route("/analytics-backend/textRanking", methods=['POST'])
def textRanking():
    """
        Text Ranking
        Get a the id of the bigram and w2v models, the list of tweet messages or the url of the csv with tweet messages and return the text ranking
        ---
        parameters:
          - in: body
            name: body
            schema:
              type: object
              properties:
                url_input:
                  type: string
                  description: url of csv with tweet messages
                bigram_model_id:
                  type: string
                  description: id of bigram model
                w2v_model_id:
                  type: string
                  description: id of word2vec model
                tweets:
                  type: array
                  items:
                    type: object
                    properties:
                      message:
                        type: string
                        description: tweeet message
            required: true
        responses:
          200:
            description: text ranking
            schema:
              type: array
              items:
                type: array
                items:
                  type: string
          500:
            description: Internal Server Error
            schema:
                type: object
                properties:
                    error:
                     type: string
    """
    log.info("/analytics-backend/textRanking")
    data_json = json.dumps(request.get_json(silent=True))
    data_json = json.loads(data_json)

    if 'url_input' in data_json:
        url_input = data_json["url_input"]
        df = pd.read_csv(url_input)

        # DEBUGGING
        debugging = conf.get('MAIN', 'debugging')
        if (debugging == 'True'):
            document_path_file = conf.get('MAIN', 'path_document')
            df = pd.read_csv(document_path_file, encoding='utf-8', error_bad_lines=False)
            df = df.head(100)

        input_list = df['message'].tolist()
    else:
        input_list = json.dumps(data_json["tweets"])
        input_list = pd.read_json(input_list, encoding='utf8')['message'].tolist()

    w2v_model_id = data_json["w2v_model_id"]
    filename = conf.get('MAIN', 'path_pickle_w2v_model_incr_fold') + "word2vec_" + str(w2v_model_id) + ".pickle"
    try:
        w2v_model = som_ms.load_obj(filename)
    except:
        filename = conf.get('MAIN', 'path_pickle_w2v_model_incr_fold') + "word2vec_" + str(w2v_model_id) + "_training.txt"
        return returnModelStatus(filename, w2v_model_id)

    bigram_model_id = data_json["bigram_model_id"]
    filename = conf.get('MAIN', 'path_pickle_bigram_model_incr_fold') + "bigram_" + str(bigram_model_id) + ".pickle"
    try:
        bigram_model = som_ms.load_obj(filename)
    except:
        filename = conf.get('MAIN', 'path_pickle_bigram_model_incr_fold') + "bigram_" + str(
            bigram_model_id) + "_training.txt"
        return returnModelStatus(filename, bigram_model_id)

    keywds = text_ranking_ms.extractKeywords(input_list, bigram_model)
    topics = TopicLabeler.textRanking(keywds, w2v_model)

    response = jsonify(topics)
    return response

# w2v
@app.route("/analytics-backend/trainWord2Vec", methods=['POST'])
def trainWord2Vec():
    """
        Train word 2 vec model
        Get a list of tweet message or the url of the csv with tweet messages, return the id of the model that will be trained and start in a new thread the training of the model (The training process takes hours)
        ---
        parameters:
          - in: body
            name: body
            schema:
              type: object
              properties:
                url_input:
                  type: string
                  description: url of csv with tweet messages
                tweets:
                  type: array
                  items:
                    type: object
                    properties:
                      message:
                        type: string
                        description: tweeet message
            required: true
        responses:
          200:
            description: Id trained Model
            schema:
                type: object
                properties:
                  w2v_model_id:
                    type: string
          500:
            description: Internal Server Error
            schema:
                type: object
                properties:
                    error:
                     type: string
    """
    log.info("/analytics-backend/trainWord2Vec")
    data_json = json.dumps(request.get_json(silent=True))
    data_json = json.loads(data_json)

    if 'url_input' in data_json:
        url_input = data_json["url_input"]
        df = pd.read_csv(url_input)

        # DEBUGGING
        debugging = conf.get('MAIN', 'debugging')
        if (debugging == 'True'):
            document_path_file = conf.get('MAIN', 'path_document')
            df = pd.read_csv(document_path_file, encoding='utf-8', error_bad_lines=False)
            df = df.head(100)

        input_list = df['message'].tolist()
    else:
        input_list = json.dumps(data_json["tweets"])
        input_list = pd.read_json(input_list, encoding='utf8')['message'].tolist()

    cleaned_input_list = clean_text_ms.cleanText(input_list)
    corpus = Corpus.createCorpus(cleaned_input_list)

    identifier = core.utility.utilities.getUniqueIdentifier()
    thread.start_new_thread(word2vec_ms.trainNewModelW2Vmodel, (corpus, identifier))

    response = jsonify({"w2v_model_id": identifier})
    return response

# train som
@app.route("/analytics-backend/trainSom", methods=['POST'])
def trainSom():
    """
        Train Self Organizing Map
        Get the id of the model word2vec, return the id of the model that will be trained and start in a new thread the training of the model (The training process takes hours)
        ---
        parameters:
          - in: body
            name: body
            schema:
              type: object
              properties:
                w2v_model_id:
                  type: string
                  description: id of model word2vec
            required: true
        responses:
          200:
            description: Id trained model
            schema:
                type: object
                properties:
                  som_model_id:
                    type: string
          500:
            description: Internal Server Error
            schema:
                type: object
                properties:
                    error:
                     type: string
          299:
            description: Model is still training or not trained
            schema:
                type: object
                properties:
                    warning:
                     type: string
    """
    log.info("/analytics-backend/trainSom")

    data_json = json.dumps(request.get_json(silent=True))
    data_json = json.loads(data_json)

    w2v_model_id = data_json["w2v_model_id"]


    filename = conf.get('MAIN', 'path_pickle_w2v_model_incr_fold') + "word2vec_" + str(w2v_model_id) + ".pickle"
    try:
        w2v_model = word2vec_ms.Word2Vec.load(filename)
    except:
        filename = conf.get('MAIN', 'path_pickle_w2v_model_incr_fold') + "word2vec_" + str(w2v_model_id) + "_training.txt"
        return returnModelStatus(filename, w2v_model_id)

    identifier = core.utility.utilities.getUniqueIdentifier()
    thread.start_new_thread(som_ms.trainNewModelBestSom, (w2v_model, identifier))

    response = jsonify({"som_model_id": identifier})
    return response

@app.route("/analytics-backend/trainNgram", methods=['POST'])
def trainNgram():
    """
            Train bigram model
            Get the list of messages or the url of the csv with tweet messages, return the id of the model that will be trained and start in a new thread the training of the model (The training process could take hours)
            ---
            parameters:
              - in: body
                name: body
                schema:
                  type: object
                  properties:
                    tweets:
                      type: array
                      items:
                        type: object
                        properties:
                          message:
                            type: string
                            description: tweeet message
                    url_input:
                      type: string
                      description: url of csv with tweet messages
                required: true
            responses:
              200:
                description: Id trained model
                schema:
                    type: object
                    properties:
                      som_model_id:
                        type: string
              500:
                description: Internal Server Error
                schema:
                    type: object
                    properties:
                        error:
                         type: string
    """
    log.info("/analytics-backend/trainNgram")
    data_json = json.dumps(request.get_json(silent=True))
    data_json = json.loads(data_json)

    if 'url_input' in data_json:
        url_input = data_json["url_input"]
        df = pd.read_csv(url_input)

        # DEBUGGING
        debugging = conf.get('MAIN', 'debugging')
        if (debugging == 'True'):
            document_path_file = conf.get('MAIN', 'path_document')
            df = pd.read_csv(document_path_file, encoding='utf-8', error_bad_lines=False)
            df = df.head()

        input_list = df['message'].tolist()
    else:
        input_list = json.dumps(data_json["tweets"])
        input_list = pd.read_json(input_list, encoding='utf8')['message'].tolist()

    identifier = core.utility.utilities.getUniqueIdentifier()
    thread.start_new_thread(text_ranking_ms.trainingNewModelBigram, (input_list, identifier))

    response = jsonify({"bigram_model_id": identifier})
    return response

@app.route("/analytics-backend/trainCodebookClustering", methods=['POST'])
def trainCodebookClustering():
    """
            Train model Codebook Clustering
            Get the id of SOM model, return the id of the model that will be trained and start in a new thread the training of the model (The training process could take hours)
            ---
            parameters:
              - in: body
                name: body
                schema:
                  type: object
                  properties:
                    som_model_id:
                      type: string
                      description: id of model SOM
                required: true
            responses:
              200:
                description: Id trained model
                schema:
                    type: object
                    properties:
                      som_model_id:
                        type: string
              500:
                description: Internal Server Error
                schema:
                    type: object
                    properties:
                        error:
                         type: string
              299:
                description: Model is still training or not trained
                schema:
                    type: object
                    properties:
                        warning:
                         type: string
    """
    log.info("/analytics-backend/trainCodebookClustering")
    data_json = json.dumps(request.get_json(silent=True))
    data_json = json.loads(data_json)

    som_model_id = data_json["som_model_id"]
    filename = conf.get('MAIN', 'path_pickle_som_model_incr_fold') + "som_" + str(som_model_id) + ".pickle"
    try:
        som_model = som_ms.load_obj(filename)
    except:
        filename = conf.get('MAIN', 'path_pickle_som_model_incr_fold') + "som_" + str(som_model_id) + "_training.txt"
        return returnModelStatus(filename, som_model_id)

    identifier = core.utility.utilities.getUniqueIdentifier()
    thread.start_new_thread(som_ms.trainNewModelCodebookCluster, (som_model, identifier))

    response = jsonify({"codebook_cluster_model_id": identifier})
    return response

@app.route("/prove/")
def hello():
    return render_template('hello.html')
    return html

@app.route("/analytics-backend/interactive-visualization/")
def analyticBackEnd():
    return render_template('analytic-back-end.html')
    return html


@app.route("/analytics-backend/tweetClassification", methods=['POST'])
def tweetClassification():
    """
        Tweet classification
        Get a tweet message as input, return the list of the first two most likely classes and the corresponding score
        ---
        parameters:
          - in: body
            name: body
            schema:
              type: object
              properties:
                message:
                  type: string
                  description: tweet message
            required: true      
        responses:  
          200:
            description: predicted classes
            type: object
            properties:
              first_class:
                type: object
                description: first predicted class
                properties:
                  label:
                    type: string
                    description: label of the predicted class
                  score:
                    type: number
                    description: score of the predicted class   
              second_class:
                type: object
                description: second predicted class
                properties:
                  label:
                    type: string
                    description: label of the predicted class
                  score:
                    type: number
                    description: score of the predicted class                   
          500:
            description: Internal Server Error
            schema:
                type: object
                properties:
                    error:
                     type: string   
    """
    log.info("/analytics-backend/tweetClassification")
    data_json = json.dumps(request.get_json(silent=True))
    data_json = json.loads(data_json)
    text = data_json['message']

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

    # choose class
    distance.index = distance['class']
    predicted_classes = distance['distance'].sort_values(ascending=False)[:2].index.to_list()
    scores = list(distance['distance'].sort_values(ascending=False)[:2].values)

    return jsonify({'first_class' : {'label': predicted_classes[0], 'score': round(scores[0],4)},
                    'second_class' : {'label': predicted_classes[1], 'score': round(scores[1],4)}})




if __name__ == "__main__":
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(host='0.0.0.0', port=10601, threaded=True)
