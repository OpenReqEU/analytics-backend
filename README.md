# Openreq Analytic Back End

![EPL 2.0](https://img.shields.io/badge/License-EPL%202.0-blue.svg "EPL 2.0")

This component was created as a result of the OpenReq project funded by the European Union Horizon 2020 Research and Innovation programme under grant agreement No 732463.

## Public APIs

The API is documented by using Swagger:

[Swagger documentation](https://api.openreq.eu/#/services/analytics-backend)

### Functionalities of the Analytic Back End

The microservices are useful to performs a topic extraction of the tweets addressed to “Wind 3” on Twitter. The algorithms allows to identify the topics of major interest and to understand what a great amount of tweets talk about, giving the possibility to pinpoint inconveniences, system failures, dissatisfaction or customer’s necessities.

It is also exposed a web interface available [here] (http://217.172.12.199:10601/openReq/interactive-visualization/).

[You can use model id 1 to test the API]

The following technologies are used:

* Flask
* pandas
* numpy
* gensim
* spaCy
* nltk

## How to Install

This microservice is Dockerized. With Docker installed on your machine, download the repo and build the project.

In order to make the component working download the file [stopwords-it.txt] (https://raw.githubusercontent.com/stopwords-iso/stopwords-it/master/stopwords-it.txt) and copy it into the folder stopwords-it

### Build docker openreq

Go inside the main folder and generate the openreq container via sudo "docker build . -t analytic-backend."

## How to Use

Call the services exposed to make text mining on the tweets, such as clean text of tweets, extract topics and graph analysis, extract keywords, apply word embedding, evaluate the SOM model:

- cleanText: get a list of tweet message, return a list of cleaned messages
- getEmbeddedWords: Get a the id of word2vec model the list of tweet messages and return a list of vector
- doSomAndPlot: Apply SOM and plot result of codebook MST
- computeTopics: Extracts topics from a list of tweets
- getCodebookActivation: Plot result of codebook Activation
- getUmatrix: Plot result of umatrix
- getCostOfSom: Get cost of Som
- keywordsExtraction: Keywords Extraction
- textRanking: Text Ranking

Here you can find the [Swagger documentation](https://api.openreq.eu/#/services/analytics-backend)

## Notes for developers

None

## How to contribute

See OpenReq project contribution
[Contribution Guidelines](https://github.com/OpenReqEU/OpenReq/blob/master/CONTRIBUTING.md)

## Sources

##### Literature used in creating some of the algorithms

- T.Kohonen: The Self-Organizing Map. http://sci2s.ugr.es/keel/pdf/algorithm/articulo/1990-
Kohonen-PIEEE.pdf.

- T. Kohonen,Self-organizing maps , Third edition.. ed. Berlin ; New York, Berlin ; New York : Springer, 2001.

- T. Kohonen: Self-Organized Formation of Topologically Correct Feature Maps. Biological Cybernetics, 1982, https://cioslab.vcu.edu/alg/Visualize/kohonen82.pdf.

- A. A. Akinduko,Principal E. M. Mirkes: Components Versus Initialization of Self-Organizing Maps: Random Initialization. A Casetudy.
https://arxiv.org/pdf/1210.5873.pdf.

- M. Attik, L. Bougrain, F. Alexandre: Self-organizing Map Initialization Artificial Neural Networks: Biological Inspirations: Lecture Notes in Computer Science, 2005.

- A Study of Parallel Self-Organizing Map Li Weigang Department of Computer Science - CIC University of Brasilia - UnBC.P 4466, CEP: 70919-970, Brasilia - DF, Brazil

## License

Free use of this software is granted under the terms of the EPL version 2 [EPL 2.0] https://www.eclipse.org/legal/epl-2.0/
