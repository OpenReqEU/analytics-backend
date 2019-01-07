import sys
sys.path.append('../..')

from ASOM import ASOM
from sklearn import datasets
import time

import numpy as np
import pylab
import somoclu


data = np.array( [[0., 0., 0.],
                 [0., 0., 1.],
                 [0., 0., 0.5],
                 [0.125, 0.529, 1.0],
                 [0.33, 0.4, 0.67],
                 [0.6, 0.5, 1.0],
                 [0., 1., 0.],
                 [1., 0., 0.],
                 [0., 1., 1.],
                 [1., 0., 1.],
                 [1., 1., 0.],
                 [1., 1., 1.],
                 [.33, .33, .33],
                 [.5, .5, .5],
                 [.66, .66, .66]])

# store the names of the data for visualization later on
color_names =    ['black', 'blue', 'darkblue', 'skyblue','greyblue', 'lilac', 'green', 'red','cyan', 'violet', 'yellow', 'white','darkgrey', 'mediumgrey', 'lightgrey']

som = ASOM(alpha_max=0.3, alpha_min=0.0001, height=3, width=3, outlier_unit_threshold = 0.5)#, cluster_model=None)

for k in range(5):
    som.train_batch(data, num_epoch=100, batch_size=None, verbose=2,)
# som.train_batch_theano(data, num_epoch=100, batch_size=None, verbose=0)
clusters = som.data2unit #lista di appartenenza delle istances  (xk in X) ai relativi som units (uh in som_unit)
# distances = som.data2dist  #lista delle distanze   delle istances (xk in X) ai relativi som units (uh in som_unit:  dist(xk,uh))

data2unit, data2cell, data2dist, data2saliency, data2saliency_index, data2maps = som.predict()
for k in range(data2dist.shape[0]):
    print "data =",data[k]
    print "map =",data2maps[k]
    print "cluster=%d  distance=%f" % (data2unit[k], data2dist[k])
    print





som.plot_units_clusters_outlier(plot_type=2)
som.plot_activations()
som.plot_mapping()





pylab.show()
