# Openreq Analytic Back End

This plugin was created as a result of the OpenReq project.

The following technologies are used:

* Flask
* pandas
* numpy
* gensim
* spaCy
* nltk

# Public APIs

The API is documented by using Swagger:

[Swagger documentation](http://217.172.12.199:10601/openReq/apispec_1.json)

## Functionalities of the Analytic Back End

The microservices are useful to performs a topic extraction of the tweets addressed to “Wind 3” on Twitter. The algorithms allows to identify the topics of major interest and to understand what a great amount of tweets talk about, giving the possibility to pinpoint inconveniences, system failures, dissatisfaction or customer’s necessities.

It is also exposed a web interface available [here] (http://217.172.12.199:10601/openReq/interactive-visualization/).

[You can use model id 1 to test the API]
