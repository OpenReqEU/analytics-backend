#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- coding: cp1252 -*-


import sys
sys.path.append('../..')

from ASOM import ASOM
from   sklearn.cluster import MiniBatchKMeans


from pandas import read_csv, DataFrame, Series, concat
from sklearn.preprocessing import MinMaxScaler

import numpy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
from matplotlib import cm
import pandas as pd

def get_train_data():

    print 'Get train data...'
    data = read_csv('./train.csv')
    data = data.drop(['Id'], axis = 1)

    # удаляем столбец Wilderness_Area2
    data = data.drop(['Wilderness_Area2', 'Vertical_Distance_To_Hydrology', 'Slope'], axis = 1)
    # data = data.drop(['Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm'], axis = 1)

    # удаляем столбцы SoilType1,...,SoilType40
    drop_soil_type_cols = []
    for k in range(1, 41):
        cname = 'Soil_Type%s' % k
        drop_soil_type_cols.append(cname)
    data = data.drop(drop_soil_type_cols, axis = 1)

    return data

def get_test_data():

    print 'Get test data...'
    data = read_csv('./test.csv')
    result = DataFrame(data.Id)

    # удаляем столбцы Id, Wilderness_Area2
    data = data.drop(['Id', 'Wilderness_Area2', 'Vertical_Distance_To_Hydrology', 'Slope'], axis = 1)
    # data = data.drop(['Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm'], axis = 1)

    # удаляем столбцы SoilType1,...,SoilType40
    drop_soil_type_cols = []
    for k in range(1, 41):
        cname = 'Soil_Type%s' % k
        drop_soil_type_cols.append(cname)
    data = data.drop(drop_soil_type_cols, axis = 1)

    return (data, result)




data = get_train_data()
test, result = get_test_data()
test = test.drop(['Aspect'], axis = 1)
train = data.drop(['Cover_Type', 'Aspect'], axis = 1)

train = train.values
test = test.values

sc = MinMaxScaler()
train = sc.fit_transform(train)
test = sc.transform(test)



# raw_input()


def findNumCluster(data):
    costs = []
    anomaly = []
    for k in range(15):
        som = ASOM(alpha_max=0.1, alpha_min=0.0, height=1, width=k + 1, outlier_unit_threshold=0.1,
                   outlier_percentile=80, Koutlier_percentile=2, learning_rate_percentile=0.1,
                   # cluster_model=None,
                    memory_size=None)
        som.train_batch(data, num_epoch=100, batch_size=None,verbose=2, training_type='adaptive', fast_training=False)
        costs.append(som.cost)
        data2unit, data2saliency, data2maps, data2dist = som.predict(data)
        anomaly.append(np.sum(data2saliency == False))

    costs = np.array(costs)
    dcosts = costs[1:]-costs[:-1]
    plt.figure('costs(num_units)')
    plt.plot(range(1,len(costs)+1),costs,'.-')
    plt.figure('anomaly(num_units)')
    plt.plot(range(1,len(costs)+1),anomaly,'.-')

    # plt.plot(dcosts,'.-')
    # plt.plot(ddcosts,'.-')
    plt.show()

# findNumCluster(train)



som = ASOM(alpha_max=0.1, alpha_min=0.0, height=10, width=10, outlier_unit_threshold=0.1,
           outlier_percentile=98, Koutlier_percentile=2, learning_rate_percentile=0.1,
           # cluster_model=None,
           memory_size=None)

som.train_batch(train, num_epoch=15, batch_size=None,verbose=2, training_type='adaptive', fast_training=False)

data2unit, data2cell, data2dist, data2saliency, data2saliency_prob, data2maps = som.predict(train)
print "------som.X.shape  = ", som.X.shape
print "--------------cost = ", som.cost
o = np.sum(data2saliency == False)
print "--------------Anomaly     =",o
print "--------------Anomaly_perc=",o/float(train.shape[0])

som.plot_units_clusters_outlier(train, plot_type=2, fig_title='Train')

test=test[::100]

data2unit, data2cell, data2dist, data2saliency, data2saliency_prob, data2maps = som.predict(test)
o = np.sum(data2saliency == False)
print "--------------Anomaly     =",o
print "--------------Anomaly_perc=",o/float(test.shape[0])
som.plot_units_clusters_outlier(test, plot_type=2, fig_title='Test')

som.plot_mapping()
som.plot_mapping_cluster( cluster_model=MiniBatchKMeans(n_clusters=8))
som.plot_mapping(test)


plt.show()
