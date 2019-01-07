#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('../..')

from scipy.io import arff
import pandas as pd
import numpy as np
import pylab
from sklearn import preprocessing

from ASOM import ASOM
from sklearn.metrics import accuracy_score
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as pylab
import numpy as np
from   sklearn.cluster import MiniBatchKMeans
from sklearn import metrics

from keras.layers.advanced_activations import LeakyReLU, PReLU
import hdbscan
import seaborn as sns


data, meta = arff.loadarff("./data/synth_multidim_100_000.arff")
data = pd.DataFrame(data)


# data = data.ix[:5000]



labels = data['class'].astype(int)
labels[labels != 0] = 1
del data['class']


min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
# min_max_scaler = preprocessing.StandardScaler()
# min_max_scaler = preprocessing.KernelCenterer()
np_scaled = min_max_scaler.fit_transform(data)
data_n = pd.DataFrame(np_scaled)
data_n = np.array(data_n,dtype=float)
print data_n.shape
print np.min(data_n)
print np.max(data_n)
# raw_input()
#
# pylab.plot(data_n.T, '.', color='b')
# pylab.plot(data_n[labels==1].T,'o',color='g')
# pylab.show()


if True:

    encoding_dim = 80
    input = Input(shape=(100,))
    encoded = Dense(encoding_dim, activation='relu')(input)
    # encoded = PReLU()(encoded)
    decoded = Dense(100, activation='sigmoid')(encoded)
    autoencoder = Model(input=input, output=decoded)

    encoder = Model(input=input, output=encoded)

    encoded_input = Input(shape=(encoding_dim,))
    decoder_layer = autoencoder.layers[-1]
    decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))

    # autoencoder.compile(optimizer='adam', loss='mean_squared_logarithmic_error')
    autoencoder.compile(optimizer='adam', loss='mse')
    # autoencoder.compile(optimizer='adam', loss='mae')
    # autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    autoencoder.fit(data_n, data_n,
                    epochs=2500,
                    batch_size=400,#400ALTO
                    shuffle=True,
                    verbose=1)

    encoded = encoder.predict(data_n)
    decoded = decoder.predict(encoded)


    dist = np.zeros(len(data_n))
    for i, x in enumerate(data_n):
        dist[i] = np.linalg.norm(x-decoded[i])


    fpr, tpr, thresholds = roc_curve(labels, dist)
    roc_auc = auc(fpr, tpr)

    pylab.figure(figsize=(10,6))
    pylab.plot(fpr, tpr, color='red', label='AUC = %0.2f)' % roc_auc)
    pylab.xlim((0,1))
    pylab.ylim((0,1))
    pylab.plot([0, 1], [0, 1], color='navy', linestyle='--')
    pylab.xlabel('False Positive rate')
    pylab.ylabel('True Positive rate')
    pylab.title('ROC Autoencoder 100-80-100 ReLU/Sigmoid synth\_multidim\_100\_000')
    pylab.legend(loc="lower right")
    pylab.show()



if True:
    hdbs = hdbscan.HDBSCAN(min_cluster_size=8, prediction_data=True, allow_single_cluster=False,min_samples=4)

    labels_prev = hdbs.fit_predict(data_n)

    pal = sns.color_palette('deep', len(np.unique(labels_prev)))
    colors = [sns.desaturate(pal[col], sat) for col, sat in zip(hdbs.labels_,hdbs.probabilities_)]
    pylab.scatter(data_n.T[0], data_n.T[1], c=colors);
    print "ACCURACY_SCORE=",accuracy_score(labels,hdbs.outlier_scores_>0.5)

    data2anomaly = hdbs.outlier_scores_>0.06
    pylab.figure()
    pylab.plot( hdbs.outlier_scores_)
    pylab.show()
    fpr, tpr, thresholds = roc_curve(labels,  hdbs.outlier_scores_)
    roc_auc = auc(fpr, tpr)

    pylab.figure(figsize=(10,6))
    pylab.plot(fpr, tpr, color='red', label='AUC = %0.2f)' % roc_auc)
    pylab.xlim((0,1))
    pylab.ylim((0,1))
    pylab.plot([0, 1], [0, 1], color='navy', linestyle='--')
    pylab.xlabel('False Positive rate')
    pylab.ylabel('True Positive rate')
    pylab.title('ROC Autoencoder 100-80-100 ReLU/Sigmoid synth\_multidim\_100\_000')
    pylab.legend(loc="lower right")
    pylab.show()


    #CONFUSION-MATRIX
    score_VS = metrics.accuracy_score(labels,data2anomaly)
    print( ("accuracy(VS):   %0.3f" % score_VS) )
    print(  metrics.classification_report(labels, data2anomaly) )
    m_VS =  metrics.confusion_matrix(labels, data2anomaly)
    print(  "confusion_matrix = ", m_VS )
    #





if False:
    som = ASOM(alpha_max=0.1, alpha_min=0.001, height=6, width=6, outlier_unit_threshold=0.01,
               outlier_percentile=95., Koutlier_percentile=1., learning_rate_percentile=0.1,
               memory_size=None)


    som.train_batch(data_n, num_epoch=2000, batch_size=None,verbose=2)


    data2unit, data2cell, data2dist, data2saliency, data2saliency_index, data2maps = som.predict(data_n)
    o = np.sum(data2saliency == False)
    r = np.sum(labels == 1)
    print "--------------anom_veri     =",r
    print "--------------Anomaly     =",o
    print "--------------Anomaly_perc=",o/float(data_n.shape[0])


    N, M=data_n.shape
    rng = range(M)

    data2anomaly=data2saliency == False
    print "ACCURACY_SCORE=",accuracy_score(labels,data2anomaly)


    som.plot_units_clusters_outlier(data_n, plot_type=2)
    # print labels
    # print data2saliency
    pylab.figure()
    g = data_n[labels==0]
    a = data_n[labels==1]


    pylab.plot(data_n[50].T, '.-', color='r')
    pylab.plot(som.W[som.unit2saliency==True].T, '.-', color='b')

    pylab.figure()
    # pylab.plot(g[:,30],g[:,41], '.', color='g')
    # pylab.plot(a[:,30],a[:,41],'o', color='r')
    pylab.plot(data2saliency_index, '.', color='k')


    # data['labels'] = labels
    # data['dist'] = data2saliency_index
    #
    # pylab.figure(figsize=(10,7))
    # pylab.scatter(data.index, data['dist'], c=data['labels'], edgecolor='black', s=17)
    # pylab.xlabel('Index')
    # pylab.ylabel('Score')
    # pylab.xlim((0,1000))
    # pylab.title("Outlier Score")
    pylab.show()




    fpr, tpr, thresholds = roc_curve(labels,  -data2saliency_index)
    roc_auc = auc(fpr, tpr)
    #
    pylab.figure(figsize=(10,6))
    pylab.plot(fpr, tpr, color='red', label='AUC = %0.2f)' % roc_auc)
    pylab.xlim((0,1))
    pylab.ylim((0,1))
    pylab.plot([0, 1], [0, 1], color='navy', linestyle='--')
    pylab.xlabel('False Positive rate')
    pylab.ylabel('True Positive rate')
    pylab.title('ROC Autoencoder 100-80-100 ReLU/Sigmoid synth\_multidim\_100\_000')
    pylab.legend(loc="lower right")
    pylab.show()

    #CONFUSION-MATRIX
    score_VS = metrics.accuracy_score(labels,data2anomaly)
    print( ("accuracy(VS):   %0.3f" % score_VS) )
    print(  metrics.classification_report(labels, data2anomaly) )
    m_VS =  metrics.confusion_matrix(labels, data2anomaly)
    print(  "confusion_matrix = ", m_VS )
    #


    som.plot_mapping()
    som.plot_mapping_cluster( cluster_model=MiniBatchKMeans(n_clusters=8))
    som.plot_mapping(data_n)
    print





pylab.show()
