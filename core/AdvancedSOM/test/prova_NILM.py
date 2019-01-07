#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- coding: cp1252 -*-


import numpy as np
from hdbscan import HDBSCAN, approximate_predict, membership_vector
from sklearn.decomposition  import  PCA, RandomizedPCA
from sklearn.manifold       import  MDS, LocallyLinearEmbedding, SpectralEmbedding
from AdvancedSOM.SilhouetteStatistics import findKopt
from sklearn.preprocessing import MinMaxScaler
from AdvancedSOM import ASOM
from AdvancedSOM.utilityAdvancedSOM import serializator
from AdvancedSOM.HDBSCAN4ASOM import HDBSCAN4ASOM
import pylab






def prova_NILM():
    from sklearn.datasets import make_moons

    som = ASOM(alpha_max=0.2, alpha_min=0.01, height=4, width=4, outlier_unit_threshold=0.01, outlier_percentile=99., Koutlier_percentile=2.)
    hdb = HDBSCAN4ASOM(min_cluster_size=5, min_samples=5, prediction_data=True, gen_min_span_tree=True)

    # X = serializator.load_obj("./data/pel3days.bin")
    X, y = make_moons(200, noise=.05, random_state=0)
    # X = serializator.load_obj("./data/dRR_indian")

    # X = min_max_scaler.fit_transform(X)
    if True:
        hdb.fit(X)
        labels = hdb.labels_
        print "NUM_CLUSTER=", len(np.unique(labels))
        hdb.plot_cluster(X, hdb.labels_, 'data+anomaly',  model_features_reduction=None)
        labels = hdb.predict(X)
        pylab.show()

    som.train_batch(X, num_epoch=100, training_type = 'adaptive',verbose=1, batch_size=100)

    som.fit_cluster(cluster_model=None,
                    perc_subsampling=60.,
                    default_cluster_model=2,
                    num_cluster_min=20,
                    num_cluster_max=20,
                    num_cluster_step = 1,
                    make_figure=1, )
    som.plot_mapping_cluster( )
    som.plot_cellDistribution()
    som.plot_units_clusters_outlier(model_features_reduction=None)#model_features_reduction=PCA(n_components=2))
    som.plot_activations()
    print som.cluster_model_centers
    print len(som.cluster_model_centers)







def main():
    prova_NILM()
    pylab.show()
    return 0



if __name__ == '__main__':
    main()




