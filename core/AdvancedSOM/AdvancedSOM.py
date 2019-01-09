#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- coding: cp1252 -*-



'''
    ASOM : Advanced Self Organizing Map for:
        1)Cluster detection
        2)Outliers detection in the cluster level
    Includes Batch and Stochastic learning rules.
    There are two different implementations:
        1)Based on Numpy
        2)Based on Theano (for big data)
    and two different training type:
        1)Online training
        1)Batch training
'''



from   __future__  import division
import warnings
from   math import *

import scipy
import numpy as np
from   sklearn.cluster import MiniBatchKMeans, KMeans
from   sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder
from   utilityAdvancedSOM.pyLogger   import  pylog
import pylab
import itertools
import time
from scipy.spatial import distance
from GapStatistics import GapStatisticsKopt
import SilhouetteStatistics
from sklearn.cluster import MeanShift,  estimate_bandwidth
# import seaborn
from sklearn.decomposition  import  PCA
from sklearn.manifold       import  MDS, LocallyLinearEmbedding, SpectralEmbedding
from SilhouetteStatistics import findKopt
from HDBSCAN4ASOM import HDBSCAN4ASOM
from hdbscan import HDBSCAN, approximate_predict, membership_vector, all_points_membership_vectors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


try:
    import theano
    import theano.tensor as tensor
    from theano import function, config, shared, sandbox
except ImportError:
    pylog.warn('Theano Import Error')




warnings.filterwarnings("ignore", category=DeprecationWarning)





DEBUG_SOM           = False
EPOCH_NUM_UNITS_MAX = 25*25*200
VERBOSE_INTERVAL    = 1







class ASOM:

    unit_x = lambda self, index, width : index % width
    unit_y = lambda self, index, width : np.floor( index / width )

    def __init__(self,  num_units                               =   25,
                        height                                  =   None,
                        width                                   =   None,
                        memory_size                             =   None,
                        alpha_max                               =   0.1,
                        alpha_min                               =   0.001,
                        count_activations                       =   True,
                        detect_outlier_unit                     =   True,
                        detect_outlier_data                     =   True,
                        outlier_unit_threshold                  =   0.2,
                        outlier_percentile                      =   90.,
                        Koutlier_percentile                     =   1.5,
                        learning_rate_percentile                =   0.1,
                        initialize_unit_with_cluster            =   1,
                        initialize_unit_with_cluster_quantile   =   0.05,
                        distance_metric                         =   'euclidean',
                        radius                                  =   None,
                 ):
        '''
        num_units:
            number of som units in som_units
            This can be changed after 2D lattice shape is computed by eigen heuristic, if its shape paramters height/width are None
        height:
            height of the 2D lattice of ASOM
        width:
            width of the 2D lattice of ASOM. height * width = num_units
        alpha_max:
            maximum learning rate that is gradually decreasing up to alpha_min
        alpha_min:
            minimum learning rate attined at the last epoch
        count_activations:
            If True => count the activation of each unit:   uk in som_units =>  activations[k]:=Nwinnings[w]/Ntot
        detect_outlier_unit:
            If True => outlier units are detected. If a unit is detected as outlier, all of the assigned items (input xk) are signed as outlier.
        detect_outlier_data:
            If True => outlier on input data are detected
        outlier_unit_threshold:
            som_units ui is saliency[outlier]  <=>
            percentuale di attivazione di ui pesata sul numero di dati appartenenti ad ui >=[<] equi-probabilita nominale(=1/num_units) * outlier_unit_threshold
        outlier_percentile:
            For each soom_unit uk, is the percentile of distances Xk:={xkh| xkh belongs to uk} to uk to detect outliers in data
                for uk in som_units:
                    outlier_data_percentile[k] := scoreatpercentile(Dk, outlier_percentile)   where   Dk:={dkh:=dist(uk,xkh) | xkh belongs to uk}    (dkh>=0)
        Koutlier_percentile:
            Koutlier_percentile * outlier_data_percentile is distance threshold to detect outliers in data:
                if xkh in Xk  and  d(xkh,uk) > Koutlier_percentile*outlier_data_percentile  =>  xkh is outlier
                Ad es. la soglia  2*median(Dk) lascia a sinistra tutti j campioni Xk     (median(Dk)=scoreatpercentile(Dk, outlier_percentile=50.0))
        learning_rate_percentile:
            Learning speed between different trainings (es. train_batch(...)) to update the outlier_percentile:
                learning_rate_percentile(k+1) = learning_rate_percentile(k) +  learning_rate_percentile*dlearning_rate_percentile(k+1)

        distance_metric : str or callable, optional
            The distance metric to use.  If a string, the distance function can be
            'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation',
            'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'kulsinski',
            'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao',
            'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean',
            'wminkowski', 'yule'.

        Attributes
        ----------
        data2saliency_index:
            if x[i] belongs to uk   =>   data2saliency_index[i] := 1.0 - dist(x[i], uk)/(Koutlier_percentile * outlier_data_percentile[i])
        data2saliency:
            if x[i] belongs to uk:
                data2saliency[i]=False   <=>   data2saliency_index[i]<0    OR    uk is anomaly
                data2saliency[i]=True    <=>   data2saliency_index[i]>0    AND   uk is saliency
        '''
        #Lattice:
        self.num_units                              =  num_units
        self.height                                 =  height
        self.width                                  =  width
        self.memory_size                            =  memory_size
        # optimization parameters
        self.alpha_max                              =  alpha_max
        self.alpha_min                              =  alpha_min
        # unit statistics
        self.count_activations                      =  count_activations
        self.detect_outlier_unit                    =  detect_outlier_unit
        self.detect_outlier_data                    =  detect_outlier_data
        self.outlier_unit_threshold                 =  outlier_unit_threshold
        self.outlier_percentile                     =  outlier_percentile
        self.Koutlier_percentile                    =  Koutlier_percentile
        self.learning_rate_percentile               =  learning_rate_percentile
        self.initialize_unit_with_cluster           =  initialize_unit_with_cluster
        self.initialize_unit_with_cluster_quantile  =  initialize_unit_with_cluster_quantile

        #start:
        self.initialized                =   False
        self.Ndata                      =   None
        self.Nfeatures                  =   None
        self.num_epoch                  =   None
        self.U                          =   None
        self.is_trained                 =   False
        self.outlier_data_percentile    =   None
        self.W                          =   None
        self.cluster_model              =   None
        self.distance_metric            = distance_metric
        self.radius                     = radius


    def buildSOM(self, X, verbose=0):
        """
            X --- data matrix with shape nxm:
                Ndata     = number of samples
                Nfeatures = number of features
        """
        if self.Nfeatures is not None  and  self.Nfeatures != X.shape[1]:
            self.initialized = False

        #Set new input:
        Ndata, Nfeatures = X.shape
        if  self.Nfeatures is not None    and   \
            self.Nfeatures == Nfeatures   and   \
            self.memory_size is not None  and   \
            self.memory_size > Ndata:
                self.X = np.vstack((self.X, X))
                self.X = self.X[-self.memory_size:]
        else:
            self.X = X
        self.Ndata, self.Nfeatures = self.X.shape

        #Reset only data statistics:
        self.data2saliency = np.array(())
        self.data2dist = np.array(())
        self.data2unit = np.array(())

        #Set som_units:
        if self.initialized:
            return None
        # raw_input('building ASOM.....')

        if (self.height is None or self.width is None) and self.num_units is not None:
            self.height = self.width  =  int( np.ceil( np.sqrt(self.num_units) ) )
            self.num_units = self.height * self.width

        if self.height is None or self.width is None:
            self._estimate_map_shape()
            self.num_units = self.height * self.width

        if self.height * self.width != self.num_units:
            pylog("Number of units is not conforming to lattice size so it is set num_units = width + heigth")
            self.num_units = self.height * self.width
            pylog("New number of units : ", self.num_units)
            # raw_input("Press Enter to continue...")

        self.data_dim = self.X.shape[1]

        # normalize data and save mean and std values?
        self.data_mean = 0
        self.data_std  = 0
        #self._norm_data()

        if self.initialize_unit_with_cluster  and  self.X is not None:
            if  self.X.shape[0]>self.num_units:
                if  self.initialize_unit_with_cluster==1:
                    cm = MiniBatchKMeans( n_clusters          =   self.num_units,
                                          init                =   'k-means++',
                                          max_iter            =   100,
                                          batch_size          =   100,
                                          verbose             =   1,
                                          compute_labels      =   False,
                                          random_state        =   None,
                                          tol                 =   0.0,
                                          max_no_improvement  =   20,
                                          init_size           =   None,
                                          n_init              =   10,
                                          reassignment_ratio  =   0.01 )
                else:
                    # #Init:
                    # self.W = np.random.random((self.num_units , self.data_dim))
                    # self.W = np.array([v/np.linalg.norm(v) for v in self.W]) #normalization
                    #Clustering:
                    quantile = self.initialize_unit_with_cluster_quantile
                    trained = False
                    while not trained:
                        try:
                            cm = MeanShift( bandwidth=estimate_bandwidth(self.X,
                                                                         quantile=quantile,
                                                                         n_samples=500),
                                            seeds=None,
                                            bin_seeding=False,
                                            min_bin_freq=1,
                                            cluster_all=True,
                                            n_jobs=1 )
                            trained = True
                            if self.num_units < cm.cluster_centers_.shape[0]:
                                self.num_units = cm.cluster_centers_.shape[0]
                                print 'New num of units for meanShift: ', self.num_units
                        except:
                            quantile += 0.005

                # Init:
                np.random.seed(int(time.time()))
                self.W = np.random.random((self.num_units, self.data_dim))
                # self.W = np.array([v / np.linalg.norm(v) for v in self.W])  # normalization
                cm =  cm.fit(self.X)
                pylog("-----------Units initialized with cluster model  =>  use training ITERATIVE!!!-----------")
                C  = cm.cluster_centers_
                # print "cm.cluster_centers_=",cm.cluster_centers_
                Nc = C.shape[0]
                #Sort initialization
                M_dist = distance.cdist(C, C, metric=self.distance_metric)
                diagonal_inf = np.vectorize(lambda x: x + np.inf if x == 0 else x)
                M_dist = diagonal_inf(M_dist)
                sorted_centers = [C[0]]
                row_index = 0
                while len(sorted_centers) < C.shape[0]:
                    min_index_row = M_dist[row_index].argmin()
                    sorted_centers.append(C[min_index_row])
                    np.delete(M_dist, min_index_row, axis=1)
                    row_index = min_index_row

                C = np.array(sorted_centers)

                if  Nc<=self.num_units:
                    self.W[:Nc] = C
                else:
                    self.W = C[:self.num_units]
                pylog("self.num_units", self.num_units)
                pylog( "Wstart = ",self.W)
                pylog(self.W.shape)
        else:
            self.W = np.random.random((self.num_units , self.data_dim))
            self.W = np.array([v/np.linalg.norm(v) for v in self.W]) #normalization

        #Set unit statistics:
        self.activations = np.zeros((self.num_units))
        self.unit2saliency_coeffs = np.zeros((self.num_units))
        self.unit2saliency = np.ones((self.num_units), dtype=bool)
        self.data2saliency = np.array(())
        self.data2saliency_index = np.array(())
        self.data2unit = np.array(())
        self.data2dist = np.array(())
        self.unit_coher = np.array(())

        #unit2cell/cell2unit
        self.u2c = {}
        self.c2u = {}
        for unit_id in range(self.num_units):
            i,j = int(self.unit_x(unit_id, self.width)),  int(self.unit_y(unit_id, self.width))
            self.u2c[unit_id]  =  (i, j)
            self.c2u[i,j] = unit_id

        #Set dist map:
        self.dist_map = np.zeros((self.num_units, self.num_units))
        for u in range(int(self.num_units)):
            self.dist_map[u,:] = self.find_neighbors(u,self.num_units)

        #Set dist data percentile:
        self.outlier_data_percentile = -np.ones(self.num_units)*2000

        #End:
        self.initialized = True

        if verbose>=2:
            self.plot_units(self.X, plot_data=True, color_data=False, title=str(self))


    def unit2cell(self, index):
        return  self.u2c[index]

    def cell2unit(self, i, j):
        return  self.c2u[i,j]


    # Euclidean distance with pre-computed data square X2
    def _euq_dist(self, X2, X):
        if self.distance_metric == 'euclidean':
            return -2*np.dot(self.W, X.T) + (self.W**2).sum(1)[:, None] + X2.T
        else:
            return distance.cdist(self.W, X, metric=self.distance_metric)



    def estimate_cost(self, X2, n_neighbors=1):
        """
           cost  =  SUM {  min(dist(W[i], X))  }
           where:
                D[i,j] = dist(W[i], X[j]) => shape=(self.num_units, X.shape[0])
        """
        D = self._euq_dist(X2, self.X)
        if n_neighbors==-1:
            cost= np.linalg.norm(D.min(0), ord=1) / self.X.shape[0]
        else:
            cost = []
            for k in range(n_neighbors):
                m    = []
                for (i,j) in enumerate(D.argmin(0)):
                    m.append(D[j,i]) #valore minimo j-mo codebook i-mo dato
                    D[j,i] = np.inf
                cost_k = np.linalg.norm(m, ord=1)/self.X.shape[0]
                cost.append(cost_k)
            # pylab.figure()
            # pylab.plot(cost,'.-')
            # pylab.show()
            cost = np.mean(cost)

        return cost

    def estimate_cost2(self, X, n_neighbors=1):
        """
           cost  =  SUM {  min(dist(W[i], X))  }
           where:
                D[i,j] = dist(W[i], X[j]) => shape=(self.num_units, X.shape[0])
        """
        X2 = (X ** 2).sum(1)[:, None]

        D = self._euq_dist(X2, X)
        if n_neighbors==-1:
            cost= np.linalg.norm(D.min(0), ord=1) / self.X.shape[0]
        else:
            cost = []
            for k in range(n_neighbors):
                m    = []
                for (i,j) in enumerate(D.argmin(0)):
                    m.append(D[j,i]) #valore minimo j-mo codebook i-mo dato
                    D[j,i] = np.inf
                cost_k = np.linalg.norm(m, ord=1)/self.X.shape[0]
                cost.append(cost_k)
            # pylab.figure()
            # pylab.plot(cost,'.-')
            # pylab.show()
            cost = np.mean(cost)

        return cost

    def _print_plot_cost(self, X2, epoch, num_epoch, verbose=0):
        cost = self.estimate_cost(X2)
        pylog("epoch", epoch, "of", num_epoch, " cost: ", cost)
        if verbose>1:
            pylab.figure('cost')
            pylab.scatter(epoch,cost,c='b')


    def set_learning_params(self, num_epoch=None, alpha_max=None, alpha_min=None):
        '''
            Before starting to learning, all imperative parameters are set regarding corresponding epoch.
            It wastes some additional memory but proposes faster learning speed.
            Outputs:
                U['alphas']   -- learning rates for each epoch
                U['H_maps']   -- matrix array of neighboorhood masks
                U['radiuses'] -- neighboor radiuses for each epoch
        '''
        if num_epoch is None:
            num_epoch = self.num_epoch
        if alpha_max is None:
            alpha_max = self.alpha_max
        if alpha_min is None:
            alpha_min = self.alpha_min

        if num_epoch*self.num_units > EPOCH_NUM_UNITS_MAX:
            pylog.warn("num_epoch*num_units>=%d  =>   Too much memory required!!!  =>  try: gen_learning_params" % EPOCH_NUM_UNITS_MAX)
            return None

        #OLD configuration:
        if  self.U is not None  and  num_epoch == self.num_epoch  and  alpha_max == self.alpha_max  and  alpha_min == self.alpha_min:
            #Return configurazione precedente:
            return self.U

        #NEW configuration:
        U = {'alphas':[], 'H_maps':[], 'radiuses':[]}
        alphas = [None]*num_epoch
        H_maps = [None]*num_epoch
        radiuses = [None]*num_epoch

        if self.radius is None:
            r0 = np.ceil(1 + floor(min(self.width, self.height)-1)/2)
            if self.initialize_unit_with_cluster:
                r0 = int(r0*0.4) + 1.
            if self.is_trained:
                r0 = int(r0*0.4) + 1.
        else:
            r0 = self.radius

        for epoch in range(0,num_epoch,1):
            alpha  = (alpha_max - alpha_min)*(num_epoch - epoch)/num_epoch   +   alpha_min
            radius = ceil(r0 * (num_epoch - epoch)/(num_epoch - 1))-1
            if radius < 0 :
                radius = 0
            neigh_updt_map = alpha * (1 - self.dist_map/float((1 + radius)))
           # neigh_updt_map[dist_map == 0] = 1
            neigh_updt_map[self.dist_map > radius] = 0 # Optimize this part
            H_maps[epoch] = neigh_updt_map
            alphas[epoch] = alpha
            radiuses[epoch] = radius
        #New configuration:
        U['alphas'] = alphas
        U['H_maps'] = H_maps
        U['radiuses'] = radiuses
        #Set new params:
        self.num_epoch  =  num_epoch
        self.alpha_min  =  alpha_min
        self.alpha_max  =  alpha_max
        self.U          =  U

        # pylab.figure('alphas')
        # pylab.plot(alphas)
        return U


    def gen_learning_params(self, num_epoch=None, alpha_max=None, alpha_min=None):
        '''
        Generator for learning params: slow but low memory required
        Outputs:
            U['alphas']   -- learning rates for each epoch
            U['H_maps']   -- matrix array of neighboorhood masks
            U['radiuses'] -- neighboor radiuses for each epoch
        '''
        if num_epoch is None:
            num_epoch = self.num_epoch
        if alpha_max is None:
            alpha_max = self.alpha_max
        if alpha_min is None:
            alpha_min = self.alpha_min

        if self.radius is None:
            r0 = np.ceil(1 + floor(min(self.width, self.height)-1)/2)
            if self.initialize_unit_with_cluster:
                r0 = int(r0*0.4) + 1.
            if self.is_trained:
                r0 = int(r0*0.4) + 1.
        else:
            r0 = self.radius

        for epoch in range(0,num_epoch):
            alpha  = (alpha_max - alpha_min)*(num_epoch - epoch)/num_epoch   +   alpha_min
            radius = ceil(r0 * (num_epoch - epoch)/(num_epoch - 1))-1
            if radius < 0 :
                radius = 0
            # print "radius=",radius
            neigh_updt_map = alpha * (1 - self.dist_map/float((1 + radius)))
            # neigh_updt_map[dist_map == 0] = 1
            neigh_updt_map[self.dist_map > radius] = 0 # Optimize this part

            yield neigh_updt_map, alpha, radius


    #################################################################################START-TRAINING######################################################################################

    def train_online(self, X, num_epoch, alpha_max=None, alpha_min=None, fast_training=False, verbose=0):
        '''
            Numpy based online stochastic training: each input istance is take individually and weight are updates in terms of winner neuron (Low MC)
            Generally faster than Theano version
        '''
        self.buildSOM(X)

        if num_epoch == None:
            num_epoch = 500 * self.num_units # Kohonen's suggestion

        start = time.time()
        pylog('Start train_online ...')
        U = None
        if fast_training:
            U = self.set_learning_params(num_epoch, alpha_max, alpha_min)
        if U is None:
            U = self.gen_learning_params(num_epoch, alpha_max, alpha_min)
        X2 = (self.X**2).sum(1)[:, None]
        for epoch in range(num_epoch):
            start = time.time()
            shuffle_indices = np.random.permutation(self.X.shape[0])
            if fast_training:
                update_rate = U['H_maps'][epoch]
                learn_rate  = U['alphas'][epoch]
            else:
                update_rate, learn_rate, _ = U.next()
            win_counts = np.zeros((self.num_units))
            for i in shuffle_indices:
                instance = self.X[i,:]
                D = self._euq_dist(X2[i][None,:], instance[None,:])
                BMU_indx = np.argmin(D)

                win_counts[BMU_indx] += 1
                if self.count_activations:
                    self.activations[BMU_indx] += 1

                self.W  +=  learn_rate * update_rate[...,BMU_indx,None]* (instance - self.W)
                ## Normalization is not imperative unless given input instances are normalized
                # self.W = self.W / np.linalg.norm(self.W)

            if verbose and (epoch % 1) == 0:
                self._print_plot_cost(X2, epoch, num_epoch, verbose)

            if self.detect_outlier_unit:
                self._update_unit_saliency(win_counts, update_rate, learn_rate)

        pylog("train_online: EsecutionTime=", time.time() - start)

        #Final cost
        self.cost = self.estimate_cost(X2)
        pylog("cost=", self.cost)

        # Normalize activation counts
        if self.count_activations:
            total_act = self.activations.sum()
            self.activations = self.activations / total_act

        self.assing_to_units() # final unit assignments

        if self.detect_outlier_unit:
            self._find_outlier_units()

        if self.detect_outlier_data:
            self._find_outlier_data()

        self.is_trained = True


    def train_batch(self, X,
                          num_epoch       =   None,
                          alpha_max       =   None,
                          alpha_min       =   None,
                          training_type   =   'adaptive',
                          batch_size      =   None,
                          fast_training   =   False,
                          verbose         =   0 ):
        '''
            Numpy based batch training: batch_size are taken individually and weight are updates in terms of winner neuron (High MC)
            If batch_size is None then all input data X are fed for each epoch.
            If fast_training=True => fast learning but high[low] memory usage  (for BigData  set fast_training=False)
        '''
        if training_type=='adaptive':
            if self.is_trained:
                training_type = 'iterative'
            else:
                training_type = 'onehot'

        self.buildSOM(X, verbose)

        if num_epoch == None:
            num_epoch = 500 * self.num_units # Kohonen's suggestion

        if batch_size ==  None:
            batch_size = self.X.shape[0]

        start = time.time()
        pylog('Start train_batch ...')
        U = None
        if fast_training:
            U = self.set_learning_params(num_epoch, alpha_max, alpha_min)
        if U is None:
            U = self.gen_learning_params(num_epoch, alpha_max, alpha_min)
        X2 = (self.X**2).sum(1)[:, None]
        epoch = 0
        while epoch < num_epoch:
            try:
                # pylog( 'Epoch --- ', epoch)
                if fast_training:
                    update_rate = U['H_maps'][epoch]
                    learn_rate  = U['alphas'][epoch]
                else:
                    update_rate, learn_rate, _ = U.next()
                # print "learn_rate",learn_rate
                # randomize batch order
                shuffle_indices = np.random.permutation(self.X.shape[0])
                win_counts = np.zeros((self.num_units))
                for batch_indices in  np.array_split(shuffle_indices, self.X.shape[0]/batch_size):
                    batch_data = self.X[batch_indices,:]
                    D = self._euq_dist(X2[batch_indices,:], batch_data)
                    BMU = (D==D.min(0)[None,:]).astype("float32").T

                    win_counts += BMU.sum(axis=0)
                    # pylog( "win_counts=",win_counts)
                    # pylog( BMU)
                    # pylog( win_counts)
                    # pylog( self.activations)
                    if self.count_activations:
                        self.activations += win_counts

                    # batch learning
                    A = np.dot(BMU, update_rate)
                    S = A.sum(0)
                    non_zeros = S.nonzero()[0]
                    Wnew = np.dot(A[:,non_zeros].T, batch_data) / S[non_zeros][..., None]
                    if training_type == 'onehot':
                        self.W[non_zeros, ...]  =  Wnew
                    else:
                        dW  =  np.zeros(self.W.shape)
                        dW[non_zeros, ...]  =  Wnew  -  self.W[non_zeros, ...]
                        self.W   +=  learn_rate*dW

                    # if epoch%50==0:
                    #     ir = np.random.randint(low=0,high=self.W.shape[1])
                    #     print "ir",ir
                    #     self.W[:,ir]  *=  1 + np.random.randn(self.W.shape[0])*0.001

                    # pylog( self.W)
                    # pylog()
                    # raw_input()

                    # normalize weight vector
                    ## Normalization is not imperative unless given input instances are normalized
                    # self.W = self.W / np.linalg.norm(self.W)
                    #self.W = self.W / np.linalg.norm(self.W)

                if self.detect_outlier_unit:
                    self._update_unit_saliency(win_counts, update_rate, learn_rate)

                if verbose  and  (epoch % VERBOSE_INTERVAL) == 0:
                   self._print_plot_cost(X2, epoch, num_epoch, verbose)

                epoch +=1

                if verbose>=2:
                    self.plot_units(plot_data=False, color_data=False, title=str(self))
                    # pylab.show()

            except KeyboardInterrupt:
                pylog("----------------KeyboardInterrupt-----------------")
                num_epoch = epoch+1

        pylog("train_batch: EsecutionTime=", time.time() - start)

        #Final cost
        self.cost = self.estimate_cost(X2)
        pylog("cost=", self.cost)

        # Normalize activation counts
        if self.count_activations:
            total_act = self.activations.sum()
            self.activations = self.activations / total_act

        self.assing_to_units() # final unit assignments

        if self.detect_outlier_unit:
            self._find_outlier_units()

        if self.detect_outlier_data:
            self._find_outlier_data()

        self.is_trained = True


    def train_online_theano(self, X, num_epoch = None, alpha_max=None, alpha_min=None, fast_training=True, verbose=0):
        '''
            Theano based online stochastic training: each input istance is take individually and weight are updates in terms of winner neuron (Low MC)
        '''
        self.buildSOM(X)

        warnings.simplefilter("ignore", DeprecationWarning)
        warnings.filterwarnings("ignore")

        if num_epoch == None:
            num_epoch = 500 * self.X.shape[0]

        # Symmbol variables
        X = tensor.dmatrix('X')
        WIN = tensor.dmatrix('WIN')
        H = tensor.dmatrix('H')

        # Init weights random
        W = theano.shared(self.W, name="W")
        #W = theano.shared(rng.randn(cluster_num, data.shape[1]).astype(theano.config.floatX), name="W")

        # Find winner unit
        D = (W**2).sum(axis=1, keepdims=True) + (X**2).sum(axis=1, keepdims=True).T - 2 * tensor.dot(W, X.T)
        bmu = (D).argmin(axis=0)
        dist = tensor.dot(WIN.T, X) - WIN.sum(0)[:, None] * W
        err = D.min(0).norm(1)/X.shape[0]

        update = function([X,WIN, H], outputs=err, updates=[(W, W + tensor.addbroadcast(H, 1) * dist)])
        find_bmu = function([X], bmu)

        start = time.time()
        pylog('Start train_online_theano ...')
        U = None
        if fast_training:
            U = self.set_learning_params(num_epoch, alpha_max, alpha_min)
        if U is None:
            U = self.gen_learning_params(num_epoch, alpha_max, alpha_min)
        for epoch in range(num_epoch):
            if fast_training:
                update_rate = U['H_maps'][epoch]
                learn_rate  = U['alphas'][epoch]
            else:
                update_rate, learn_rate, _ = U.next()
            win_counts = np.zeros((self.num_units))
            shuff_indx = np.random.permutation(self.X.shape[0])
            for i in shuff_indx:
                ins = self.X[i, :][None,:]
                D = find_bmu(ins)
                S = np.zeros([ins.shape[0],self.num_units])
                #S = np.zeros([batch,cluster_num], theano.config.floatX)
                S[:,D] = 1
                win_counts[D] += 1
                h = update_rate[D,:].sum(0)[:,None]
                cost = update(ins,S,h)

        pylog("train_online_theano: EsecutionTime=", time.time() - start)

        if verbose:
            pylog("Avg. centroid distance -- ", cost, "\t EPOCH : ", epoch, " of ", num_epoch)
        if self.count_activations:
            self.activations += win_counts

        if self.detect_outlier_unit:
            self._update_unit_saliency(win_counts, update_rate, learn_rate)

         # get the data from shared theano variable
        self.W = W.get_value()

        # Normalize activation counts
        if self.count_activations:
            total_act = self.activations.sum()
            self.activations = self.activations / total_act

        self.assing_to_units() # final unit assignments

        if self.detect_outlier_unit:
            self._find_outlier_units()

        if self.detect_outlier_data:
            self._find_outlier_data()

        self.is_trained = True


    def train_batch_theano(self, X, num_epoch = None, alpha_max=None, alpha_min=None, batch_size = None, fast_training=True, verbose=True):
        '''
            Theano based batch learning:  batch_size are taken individually and weight are updates in terms of winner neuron (High MC)
            If batch_size is None then all input data X are fed for each epoch.
            This Theano version is faster if data is Big.
            For BigData  =>  reduce batch_size to reduce MC
        '''

        self.buildSOM(X)

        if num_epoch == None:
            num_epoch = 500 * self.X.shape[0]

        if batch_size == None:
            batch_size = self.X.shape[0]

        # Symmbol variables
        X = tensor.dmatrix('X')
        # WIN = tensor.dmatrix('WIN')
        # alpha = tensor.dscalar('learn_rate')
        H = tensor.dmatrix('update_rate')

        # Init weights random
        W = theano.shared(self.W, name='W')
        # W_old = W.get_value()

        # Find winner unit
        D = (W**2).sum(axis=1, keepdims=True) + (X**2).sum(axis=1, keepdims=True).T - 2 * tensor.dot(W, X.T)
        BMU = (tensor.eq(D, D.min(axis=0, keepdims=True))).T
        # dist = tensor.dot(BMU.T, X) - BMU.sum(0)[:, None] * W
        err = D.min(0).sum().norm(1)/X.shape[0]

        #update = function([X,WIN,alpha],outputs=err,updates=[(W, W + alpha * dist)])

        A = tensor.dot(BMU, H)
        S = A.sum(axis=0)
        update_neigh_no_verbose = function([X, H], outputs=BMU, updates=[(W, tensor.where((S[:, None] > 0), tensor.dot(A.T, X), W) / tensor.where((S > 0), S, 1)[:, None])])
        update_neigh = function([X, H], outputs=[err, BMU], updates=[(W, tensor.where((S[:, None] > 0), tensor.dot(A.T, X), W) / tensor.where((S > 0), S, 1)[:, None])])
        # find_bmu = function([X], BMU)

        start = time.time()
        pylog('Start train_batch_theano ...')
        U = None
        if fast_training:
            U = self.set_learning_params(num_epoch, alpha_max, alpha_min)
        if U is None:
            U = self.gen_learning_params(num_epoch, alpha_max, alpha_min)
        for epoch in range(num_epoch):
            start = time.time()
            # pylog( 'Epoch --- ', epoch)
            if fast_training:
                update_rate = U['H_maps'][epoch]
                learn_rate  = U['alphas'][epoch]
            else:
                update_rate, learn_rate, _ = U.next()
            win_counts = np.zeros((self.num_units))
            for i in range(0, self.X.shape[0], batch_size):
                batch_data = self.X[i:i+batch_size, :]
                #temp = find_bmu(batch_data)
                if verbose and epoch % 5 == 0:
                    cost, winners = update_neigh(batch_data, update_rate)
                else:
                    winners = update_neigh_no_verbose(batch_data, update_rate)
                win_counts =+ winners.sum(axis=0)
                ## Normalization is not imperative unless given input instances are normalized
                # self.W = self.W / np.linalg.norm(self.W)


            if verbose and epoch % 10 == 0:
                pylog("Avg. centroid distance -- ", cost, "\t EPOCH : ", epoch, " of ", num_epoch)

            if self.count_activations:
                self.activations += win_counts

            if self.detect_outlier_unit:
                self._update_unit_saliency(win_counts, update_rate, learn_rate)

        pylog("train_batch_theano: EsecutionTime=", time.time() - start)

        # get the data from shared theano variable
        self.W = W.get_value()

        # Normalize activation counts
        if self.count_activations:
            total_act = self.activations.sum()
            self.activations = self.activations / total_act

        self.assing_to_units() # final unit assignments

        if self.detect_outlier_unit:
            self._find_outlier_units()

        if self.detect_outlier_data:
            self._find_outlier_data()

        self.is_trained = True


    def fit(self):
        pass



    #################################################################################STOP-TRAINING######################################################################################


    # Find the neighbooring units to given unit (Uses the Chessboard distance)
    def find_neighbors(self, unit_id, radius):
        neighbors = np.zeros((1,self.num_units))
        test_neig = np.zeros((self.height, self.width))
        unit_x, unit_y = self.unit2cell(unit_id)

        min_y = max(int(unit_y - radius), 0)
        max_y = min(int(unit_y + radius), self.height-1)
        min_x = max(int(unit_x - radius), 0)
        max_x = min(int(unit_x + radius), self.width-1)
        for y in range(min_y, max_y+1,1):
            for x in range(min_x, max_x+1,1):
                dist = abs(y-unit_y) + abs(x-unit_x)
                neighbors[0, x + ( y * self.width )] = dist
                test_neig[y,x] = dist
        return neighbors


    # structure the unit weight to be shown at U-matrix
    def evaluate_UMatrix(self):
        unit_xy  = np.reshape(range(self.num_units),(self.height, self.width))
        sqrt_weigths = np.reshape(self.W,(self.height, self.width, self.data_dim))
        UM = np.zeros((sqrt_weigths.shape[0],sqrt_weigths.shape[1]))
        it = np.nditer(UM, flags=['multi_index'])
        while not it.finished:
            for ii in range(it.multi_index[0]-1,it.multi_index[0]+2):
                for jj in range(it.multi_index[1]-1,it.multi_index[1]+2):
                    if ii >= 0 and ii < sqrt_weigths.shape[0] and jj >= 0 and jj < sqrt_weigths.shape[1]:
                        UM[it.multi_index] += np.linalg.norm(sqrt_weigths[ii,jj,:]-sqrt_weigths[it.multi_index])
            it.iternext()
        UM = UM/UM.max()
        self.UM = UM
        return UM, unit_xy


    # set the ratio of width and height of the map by the
    # ratio between largest 2 eigenvalues, computed from data
    def _estimate_map_shape(self):
        # pylog( self.X.shape[0])
        # raw_input()
        #num_instances = self.X.shape[0]
        u,s,v = np.linalg.svd(self.X ,full_matrices = False)
        s_sorted = np.sort(s)[::-1]
        ratio = s_sorted[0] / s_sorted[1]
        self.height = int(min(self.num_units, np.ceil(np.sqrt(self.num_units / ratio))))
        self.width = int(np.ceil(self.num_units / self.height))
        # self.height = int(np.round(np.sqrt(num_instances)))
        # self.width = int(np.round(num_instances / self.height))
        pylog('Estimated map size is -> height = ', self.height, ' width = ', self.width)


    def find_units_coherence(self):
        '''
            Find individually coherence of each unit by looking to avg. distance
            between unit weight and the assigned input instances self.X
        '''
        self.unit_coher = np.zeros((self.num_units))
        for i in np.unique(self.data2unit):
            indices = np.where(self.data2unit == i)
            self.unit_coher[i] = np.sum(self.data2dist[indices]) / indices[0].size


    # Assign input instances (xk in X) to BMUs (Best Match Units) and evaluate relative distances
    def assing_to_units(self, X=None):
        if X is None:
            X2 = (self.X**2).sum(1)[:, None]
            D = -2*np.dot(self.W, self.X.T) + (self.W**2).sum(1)[:, None] + X2.T
            self.data2unit = D.argmin(axis=0)
            self.data2dist = D[self.data2unit, np.arange(self.X.shape[0])]
            return self.data2unit, self.data2dist
        else:
            X2 = (X**2).sum(1)[:, None]
            D = -2*np.dot(self.W, X.T) + (self.W**2).sum(1)[:, None] + X2.T
            data2unit = D.argmin(axis=0)
            data2dist = D[data2unit, np.arange(X.shape[0])]
            return data2unit, data2dist


    def _find_outlier_data(self):
        '''
            Find the poor instances xhk in data self.X relative to salient units uk in som_units
            If a unit is detected as outlier, all of the assigned data (input xk) are signed as outlier data
        '''
        #1)Sign outlier data if is in  outlier_units:
        if self.data2saliency.size == 0  or   self.data2saliency_index.size!=self.X.shape[0]:
            self.data2saliency       = np.ones((self.X.shape[0]), dtype=bool)
            self.data2saliency_index = np.ones((self.X.shape[0]))
        outlier_units = np.where(self.unit2saliency == False)[0]
        for i in outlier_units:
            indices = np.where(self.data2unit == i)[0]
            self.data2saliency[indices] = False
            self.data2saliency_index[indices] = -1000.0

        #2)Sign outlier data if is over threshold:
        for i in np.unique(self.data2unit):

            indices = np.where(self.data2unit == i)[0]
            dist    =  self.data2dist[indices]
            if dist.shape[0] > 2:
                if self.outlier_data_percentile[i] < 0:
                    self.outlier_data_percentile[i] =  scipy.stats.scoreatpercentile(dist, self.outlier_percentile)
                else:
                    self.outlier_data_percentile[i] +=  (scipy.stats.scoreatpercentile(dist, self.outlier_percentile) -  self.outlier_data_percentile[i]) * self.learning_rate_percentile
            if self.outlier_data_percentile[i] > 0:
                thr = self.Koutlier_percentile * self.outlier_data_percentile[i]  +  1e-6
                self.data2saliency_index[indices] =  1.0 - dist/thr
                outlier_insts = indices[dist > thr]
                self.data2saliency[outlier_insts] = False
            else:
                thr = 0
                self.data2saliency_index[indices] =  -1000.0
                self.data2saliency[indices]       =  False

            if DEBUG_SOM:
                pylab.figure('TRAINING distances to Codebook=%d' % i)
                pylab.subplot(211)
                pylab.scatter(range(len(dist)),dist)
                pylab.axhline(self.outlier_data_percentile[i], c='b')
                pylab.axhline(thr, c='r')
                pylab.subplot(212)
                pylab.hist(dist, bins=70)
                pylab.axvline(self.outlier_data_percentile[i], c='b')
                pylab.axvline(thr, c='r')
        # pylog("outlier_data_percentile =",self.outlier_data_percentile)
        # pylog("outlier_data_threshold  =",self.Koutlier_percentile * self.outlier_data_percentile)


    # return BMU=data2unit, BMUdistance=data2dist, BUMsaliency=data2saliency by already trained params
    def process_new_data(self, X):
        data2unit, data2dist = self.assing_to_units(X)

        #1)Sign outlier data if is in  outlier_units:
        data2saliency       = np.ones((X.shape[0]), dtype=bool)
        data2saliency_index = np.ones((X.shape[0]))
        outlier_units = np.where(self.unit2saliency == False)[0]
        for i in outlier_units:
            indices = np.where(data2unit == i)[0]
            data2saliency[indices] = False
            data2saliency_index[indices] = -1000.0

        #2)Sign outlier data if is over threshold:
        for i in np.unique(data2unit):
            #Calcolo outlier sui nuovi dati X:
            indices  =  np.where(data2unit == i)[0]
            dist     =  data2dist[indices]
            if self.outlier_data_percentile[i] > 0:
                thr      =  self.Koutlier_percentile * self.outlier_data_percentile[i]  +  1e-6
                data2saliency_index[indices] =  1.0 - dist/thr
                outlier_insts = indices[dist > thr]
                data2saliency[outlier_insts] = False
            else:
                thr = 0
                data2saliency_index[indices] =  -1000.0
                data2saliency[indices]       = False

            if DEBUG_SOM:
                pylab.figure('PREDICTIONS: distances to Codebook=%d' % i)
                pylab.subplot(211)
                pylab.scatter(range(len(dist)),dist)
                pylab.axhline(self.outlier_data_percentile[i], c='b')
                pylab.axhline(thr,  c='r')
                pylab.subplot(212)
                pylab.hist(dist, bins=70)
                pylab.axvline(self.outlier_data_percentile[i], c='b')
                pylab.axvline(thr,  c='r')

        return data2unit, data2dist, data2saliency, data2saliency_index


    def predict(self, X=None):
        """
        data2unit        =>   X[k]  -----clusters------------>  C[k]
        data2saliency    =>   X[k]  -----is_normal----------->  1/0   [normal/outlier]
        data2maps        =>   X[k]  -----features_map-------->  FM[k]
        data2dist        =>   X[k]  -----features_distance--->  dist(X[k], FM[k])
        """
        if X is None:
            data2unit, data2dist, data2saliency, data2saliency_index  =   self.data2unit, self.data2dist, self.data2saliency, self.data2saliency_index
        else:
            X = np.array(X).reshape((-1,self.Nfeatures))
            data2unit, data2dist, data2saliency, data2saliency_index  = self.process_new_data(X)
        data2maps   =   np.copy(self.W[data2unit])
        data2cell = []
        for  unit_id in data2unit:
            cell = self.unit2cell(unit_id)
            data2cell.append(cell)
        data2cell = np.array(data2cell)

        return data2unit, data2cell, data2dist, data2saliency, data2saliency_index, data2maps


    def make_Xsom(self, perc_subsampling=90.0, make_figure=False):

        if self.W is None:
            pylog.warn('ASOM not trained!')
            return None

        Xsom = np.copy(self.W)
        if perc_subsampling > 0:
            for k in range(self.num_units):
                ik     = self.data2unit==k
                Xk     = self.X[ik]
                Dk     = self.data2dist[ik]
                dmax_k = scipy.stats.scoreatpercentile(Dk, perc_subsampling)
                Xsom = np.vstack((Xsom,Xk[Dk<dmax_k]))

            if make_figure>0:
                pylab.figure()
                pylab.title('make_Xsom')
                pylab.scatter(Xsom[:,0],Xsom[:,1],color='g', label='Xsom')
                pylab.legend()

        self.Xsom = Xsom


    def fit_cluster(self,   cluster_model         =   None,
                            perc_subsampling      =   20.,
                            num_cluster_min       =   None,
                            num_cluster_max       =   None,
                            num_cluster_step      =   1,
                            default_cluster_model =   1,
                            find_cluster_min      =   True,
                            make_figure           =   False,
                    ):

        if self.W is None:
            pylog.warn('ASOM not trained!')
            return None

        #Building Input matrix for clustering:
        self.make_Xsom(perc_subsampling = perc_subsampling,
                       make_figure      = make_figure-1)

        #Building cluster_model:
        cluster_model_trained = False
        if  cluster_model is None  and  self.cluster_model is None:
            if default_cluster_model==0:
                self.cluster_model =   MiniBatchKMeans(  n_clusters          =   self.num_units,
                                                         init                =   'k-means++',
                                                         max_iter            =   100,
                                                         batch_size          =   100,
                                                         verbose             =   0,
                                                         compute_labels      =   False,
                                                         random_state        =   None,
                                                         tol                 =   0.0,
                                                         max_no_improvement  =   20,
                                                         init_size           =   None,
                                                         n_init              =   10,
                                                         reassignment_ratio  =   0.01 )
                if num_cluster_min is None or num_cluster_max is None:
                    num_cluster_min  = 2
                    num_cluster_max  = int(self.num_units/2)
                    num_cluster_step = 1

            elif default_cluster_model==1:
                self.cluster_model = GaussianMixture( n_components     =   1,
                                                      covariance_type  =   'full',
                                                      tol              =   1e-4,
                                                      reg_covar        =   1e-6,
                                                      max_iter         =   500,
                                                      n_init           =   10,
                                                      init_params      =   'kmeans',
                                                      random_state     =   None,
                                                      warm_start       =   False,
                                                      verbose          =   0 )
                if num_cluster_min is None or num_cluster_max is None:
                    num_cluster_min  = 2
                    num_cluster_max  = int(self.num_units/2)
                    num_cluster_step = 1

            else:
                self.cluster_model = HDBSCAN4ASOM( min_cluster_size     =   num_cluster_min,
                                                   prediction_data      =   True,
                                                   min_samples          =   1,
                                                   metric               =   self.distance_metric,
                                                   algorithm            =   'best',
                                                   leaf_size            =   40,
                                                   approx_min_span_tree =   False,
                                                   gen_min_span_tree    =   False,
                                                   core_dist_n_jobs     =   -1,
                                                   allow_single_cluster =   False )
                if num_cluster_min is None or num_cluster_max is None:
                    num_cluster_min  = int(self.Xsom.shape[0] / self.num_units * 0.5) + 2
                    num_cluster_max  = int(num_cluster_min*2.) + 2
                    num_cluster_step = int((num_cluster_max - num_cluster_min)/10)+1
                    print  "num_cluster_min", num_cluster_min
                    print  "num_cluster_max", num_cluster_max


            optimal_k, self.cluster_model = findKopt(cluster_model    =  self.cluster_model,
                                                     X                =  self.Xsom,
                                                     num_cluster_min  =  num_cluster_min,
                                                     num_cluster_max  =  num_cluster_max,
                                                     num_cluster_step =  num_cluster_step,
                                                     use_anomaly      =  True,
                                                     find_cluster_min =  find_cluster_min,
                                                     make_figure      =  make_figure)

            if optimal_k>self.num_units:
                print "n_clusters_opt(corretto)=",self.num_units
                self.cluster_model =  MiniBatchKMeans(   n_clusters          =   self.num_units,
                                                         init                =   'k-means++',
                                                         max_iter            =   100,
                                                         batch_size          =   100,
                                                         verbose             =   0,
                                                         compute_labels      =   False,
                                                         random_state        =   None,
                                                         tol                 =   0.0,
                                                         max_no_improvement  =   20,
                                                         init_size           =   None,
                                                         n_init              =   10,
                                                         reassignment_ratio  =   0.01 )
                cluster_model_trained = False
            else:
                cluster_model_trained = True

        elif cluster_model is not None:
            self.cluster_model = cluster_model

        #Fit cluster model:
        if not cluster_model_trained:
            self.cluster_model.fit(self.Xsom)

        #Mapping units-->cluster
        self.unit2cluster = self.cluster_model.predict(self.W) + 1
        #Set cluster anomaly to <0:
        self.unit2cluster *= self.unit2saliency*2-1

        #Calcolo dei center e sigma dei cluster:
        self.cluster_model_centers = []
        self.cluster_model_sigmas  = []
        for c in np.unique(self.unit2cluster):
            Wc = self.W[self.unit2cluster == c]
            self.cluster_model_centers.append(np.mean(Wc, axis=0))
            self.cluster_model_sigmas.append(np.std(Wc, axis=0))

        self.cluster_model_centers = np.array(self.cluster_model_centers, dtype=float)
        self.cluster_model_sigmas  = np.array(self.cluster_model_sigmas, dtype=float)

        return self.cluster_model


    def predict_cluster(self, X=None):

        if self.W is None:
            pylog.warn('ASOM not trained!')
            return None

        if self.cluster_model is None:
            pylog.warn('CLUSTER MODEL not trained!')
            return None

        if X is None:
            data2unit = self.data2unit
        else:
            data2unit, data2cell, data2dist, data2saliency, data2saliency_index, data2maps = self.predict(X)

        data2cluster = self.unit2cluster[data2unit]
        return data2cluster


    def fit_predict_cluster(self,   X                       =   None,
                                    cluster_model           =   None,
                                    make_figure             =   False,
                                    steps_after_optimum     =   0,
                                    fast_mode               =   True,
                                    num_cluster_min             =   1,
                                    default_cluster_model   =   0,
                            ):

        self.fit_cluster(cluster_model          =   cluster_model,
                         make_figure            =   make_figure,
                         steps_after_optimum    =   steps_after_optimum,
                         fast_mode              =   fast_mode,
                         num_cluster_min            =   num_cluster_min,
                         default_cluster_model =   default_cluster_model,
                         )
        if X is None:
            X = self.W
        return self.predict_cluster(X, make_figure)


    def train_dweight(self, num_epoch = None,  batch_size=None):
        '''
            Numpy based batch training: batch_size are taken individually and weight are updates in terms of winner neuron (High MC)
            If batch_size is None then all input data X are fed for each epoch.
            If fast_training=True => fast learning but high[low] memory usage  (for BigData  set fast_training=False)
        '''

        pylog.info('Shape of input: ', self.X.shape)

        if num_epoch == None:
            num_epoch = 500 * self.num_units # Kohonen's suggestion

        if batch_size ==  None:
            batch_size = self.X.shape[0]

        start = time.time()
        pylog('Start train_batch ...')

        U = self.gen_learning_params(num_epoch, self.alpha_max, self.alpha_min)
        X2 = (self.X**2).sum(1)[:, None]
        epoch = 0
        while epoch<num_epoch:
            try:

                update_rate, learn_rate, _ = U.next()
                # print "learn_rate",learn_rate
                # randomize batch order
                shuffle_indices = np.random.permutation(self.X.shape[0])
                win_counts = np.zeros((self.num_units))
                for batch_indices in  np.array_split(shuffle_indices, self.X.shape[0]/batch_size):
                    batch_data = self.X[batch_indices,:]
                    D = self._euq_dist(X2[batch_indices,:], batch_data)
                    BMU = (D==D.min(0)[None,:]).astype("float32").T

                    win_counts += BMU.sum(axis=0)
                    # pylog( "win_counts=",win_counts)
                    # pylog( BMU)
                    # pylog( win_counts)
                    # pylog( self.activations)
                    if self.count_activations:
                        self.activations += win_counts

                    # batch learning
                    A = np.dot(BMU, update_rate)
                    S = A.sum(0)
                    non_zeros = S.nonzero()[0]

                    dW  =  np.zeros(self.W.shape)
                    dW[non_zeros, ...]  =  np.dot(A[:,non_zeros].T, batch_data) / S[non_zeros][..., None]  -  self.W[non_zeros, ...]

                    self.W   +=  learn_rate*dW

                    # pylog( self.W)
                    # pylog()
                    # raw_input()

                    # normalize weight vector
                    ## Normalization is not imperative unless given input instances are normalized
                    # self.W = self.W / np.linalg.norm(self.W)
                    #self.W = self.W / np.linalg.norm(self.W)

                if self.detect_outlier_unit:
                    self._update_unit_saliency(win_counts, update_rate, learn_rate)

                epoch +=1

            except KeyboardInterrupt:
                pylog("----------------KeyboardInterrupt-----------------")
                num_epoch = epoch+1

        pylog("train_batch: EsecutionTime=", time.time() - start)


    def weights_update(self, dW, dUnit2saliency_coeffs, dActivations):

        self.W                      +=  dW
        self.unit2saliency_coeffs   += dUnit2saliency_coeffs
        self.activations            += dActivations



    def updates_after_trainig(self):

        # Normalize activation counts
        if self.count_activations:
            total_act = self.activations.sum()
            self.activations = self.activations / total_act

        self.assing_to_units() # final unit assignments

        if self.detect_outlier_unit:
            self._find_outlier_units()

        if self.detect_outlier_data:
            self._find_outlier_data()

        self.is_trained = True


    def evaluate_num_units_opt(self, Kopt=5, make_figure=True):

        optimal_k = self.find_optimal_k(make_figure=make_figure)
        #Empirical model:
        num_units_opt = optimal_k*Kopt
        height = width = int( np.ceil( np.sqrt(self.num_units) ) )
        return height, width



    def transform(self, X, make_figure=False):

        #Distances from all units:
        X2 = (X**2).sum(1)[:, None]
        res = -2*np.dot(self.W, X.T) + (self.W**2).sum(1)[:, None] + X2.T
        res = res.T
        if make_figure:
            pylab.figure()
            pylab.plot(res)
        return res


    def _update_unit_saliency(self, win_counts, update_rate, learn_rate):
        '''
            It is called after each epoch of the learning.
            Compute the unit saliencies: at the end, those values defines the outlier and salient units
        '''

        excitations = (update_rate * win_counts).sum(axis=0) / learn_rate
        excitations = excitations / excitations.sum()
        single_excitations = win_counts * learn_rate
        single_excitations = single_excitations / single_excitations.sum()
        self.unit2saliency_coeffs += excitations + single_excitations
        # print "self.unit2saliency_coeffs=",self.unit2saliency_coeffs


    def _find_outlier_units(self):
        '''
            Find outlier=>salient units uk in som_units
        '''
        # find outlier units: som_units-ith:=ui is saliency  <=>
        # percentuale di attivazione di ui pesata sul numero di dati appartenenti ad ui >=[<] equi-probabilita nominale(=1/num_units) * outlier_unit_threshold
        unique, counts = np.unique(self.data2unit, return_counts=True, return_inverse=False)
        unique_perc = np.zeros(self.num_units)
        unique_perc[unique] = counts/float(np.sum(counts))
        unique_perc  =  unique_perc/np.max(unique_perc) *2   #  in (0.,2.)
        print 'unique_perc=',unique_perc
        print 'self.unit2saliency_coeffs/self.unit2saliency_coeffs.sum()=',self.unit2saliency_coeffs/self.unit2saliency_coeffs.sum()
        self.unit2saliency  = unique_perc  * self.unit2saliency_coeffs/self.unit2saliency_coeffs.sum()  >  self.outlier_unit_threshold/self.num_units


    # Returns indices of salient input instances: self.X
    def salient_data_index(self):
        return np.where(self.data2saliency == True)[0]


    def salient_unit_index(self):
        return np.where(self.unit2saliency == True)[0]


    def salient_insts(self):
        return self.X[np.where(self.data2saliency == True)]


    def salient_units(self):
        return self.W[np.where(self.unit2saliency == True)]


    def _norm_data(self, X = None):
        '''
            Take the norm of the given data matrix and save std and mean
            for future purposes
        '''

        if X is None:
            self.data_mean =  self.X.mean(axis=0)
            self.data_std  =  self.X.std(axis=0, ddof=1)
            self.X = (self.X - self.data_mean) / (self.data_std  + np.finfo(float).eps)
        else:
            data_mean =  X.mean(axis=0)
            data_std  =  X.std(axis=0, ddof=1)
            X = (X - data_mean) / data_std
            return X, data_mean, data_std


    def plot_units(self, X=None, title='', plot_data=False, color_data=False):
        if X is None:
            X         = self.X
            data2unit = self.data2unit
        else:
            data2unit = self.assing_to_units(X)[0]
        pylab.figure('plot_units: '+title)

        colors = pylab.cm.rainbow(np.linspace(0, 0.8, self.num_units))
        if plot_data:
            if color_data:
                pylab.scatter(X[:,0], X[:,1],  color=colors[data2unit])
            else:
                pylab.scatter(X[:,0], X[:,1],  color='g')
        W = self.W
        pylab.scatter(W[:,0], W[:,1],  color=colors,  s=150 ,edgecolor='none', marker='+')
        for count, i in enumerate(W):
            pylab.annotate(count, xy = i[:2], xytext = (0, 0), textcoords = 'offset points')


    def plot_activations(self):
        activations = self.activations
        pylab.figure('Codebook activations')
        sorted_idx = np.argsort(activations)
        pos = np.arange(sorted_idx.shape[0])
        activations_perc = activations[sorted_idx]
        pylab.barh(pos, activations_perc, align='center', color='b')
        pylab.yticks(pos, sorted_idx, rotation=0, fontsize=6)


    def plot_units_outlier(self, X=None):
        if X is None:
            X = self.X
            data2saliency = self.data2saliency
        else:
            data2unit, data2dist, data2saliency, data2saliency_index  = self.process_new_data(X)
        W = self.W
        unit2saliency = self.unit2saliency
        activations   = self.activations
        activations_perc = activations/activations.max()*100.0  +  35.

        pylab.figure('plot_units_outlier')
        pylab.scatter(X[data2saliency == True,  0], X[data2saliency == True,  1],  color='g',   s=50 , edgecolor='none', marker='.')
        pylab.scatter(X[data2saliency == False, 0], X[data2saliency == False, 1],  color='r',   s=50 , edgecolor='none', marker='.')
        pylab.scatter(W[unit2saliency == True , 0], W[unit2saliency == True , 1],  color='b',   s=activations_perc[unit2saliency==True].T ,edgecolor='none',  marker='+')
        pylab.scatter(W[unit2saliency == False ,0], W[unit2saliency == False ,1],  color='m',   s=activations_perc[unit2saliency==False].T ,edgecolor='none', marker='+')
        for count, i in enumerate(W):
            pylab.annotate(count, xy = i[:2], xytext = (0, 0), textcoords = 'offset points')


    def plot_units_clusters_outlier(self,   X                        =  None,
                                            plot_type                =  0,
                                            fig_title                =  '',
                                            labels                   =  None,
                                            model_features_reduction =  1,
											dimensions=2											):
        if X is None:
            X = self.X
            data2unit, data2dist, data2saliency, data2saliency_index  =   self.data2unit, self.data2dist, self.data2saliency, self.data2saliency_index
        else:
            data2unit, data2dist, data2saliency, data2saliency_index  = self.process_new_data(X)

        W = self.W
        if self.cluster_model is not None:
            Centers = self.cluster_model_centers
            data2cluster = self.predict_cluster(X)
            centers2cluster = self.predict_cluster(Centers)

        unit2saliency = self.unit2saliency
        activations = self.activations
        activations_perc = activations/activations.max()*100.0  +  35.

        if model_features_reduction is not None:
            if dimensions==2:
                
                if model_features_reduction == 0:
                    model_features_reduction = PCA(2)
                elif model_features_reduction ==1:
                    model_features_reduction = PCA(2, svd_solver='randomized')
                
                else:
                    model_features_reduction = LocallyLinearEmbedding(n_neighbors=model_features_reduction, n_components=2)
                data_matrix = np.concatenate([X, W], axis=0)
                scaling_matrix = model_features_reduction.fit_transform(data_matrix)
                X = scaling_matrix[:X.shape[0]]
                W = scaling_matrix[X.shape[0]:]
                if self.cluster_model is not None:
                    Centers = model_features_reduction.transform(Centers)

            else:
                if model_features_reduction == 0:
                    model_features_reduction = PCA(3)
                elif model_features_reduction ==1:
                    model_features_reduction = PCA(3, svd_solver='randomized')
                
                else:
                    model_features_reduction = LocallyLinearEmbedding(n_neighbors=model_features_reduction, n_components=2)
                data_matrix = np.concatenate([X, W], axis=0)
                scaling_matrix = model_features_reduction.fit_transform(data_matrix)
                X = scaling_matrix[:X.shape[0]]
                W = scaling_matrix[X.shape[0]:]
                if self.cluster_model is not None:
                    Centers = model_features_reduction.transform(Centers)
        fig = None
        if plot_type in (0,2):
			if dimensions==2:
				
				pylab.figure('plot_Codebook_Outlier' + fig_title)
				colors =  pylab.cm.Paired( np.linspace(0.1, 0.9,  self.num_units) )  #+  list( pylab.cm.gnuplot2( np.linspace(0.1, 0.9,  Nc) ) ) )[:self.num_units]
				# colors = pylab.cm.rainbow(np.linspace(0, 0.9, Nc))
				d2s=None
				if DEBUG_SOM:
					d2s= (data2saliency_index*5)**3+5
				pylab.scatter(X[:,0], X[:,1],  color=colors[data2unit], s=d2s, alpha=0.5)
				pylab.scatter(X[data2saliency == False, 0], X[data2saliency == False, 1],  color='r',   s=40 , edgecolor='none', marker='x', label='data_anomaly')

				if labels:
					for label, x, y in zip(labels, list(X[:,0]), list(X[:,1])):
						try:
							pylab.annotate(label, (x, y), size=7)
						except:
							pass

				i_ok = unit2saliency == True
				i_ko = unit2saliency == False
				pylab.scatter(W[i_ok , 0], W[i_ok , 1],  color='k',   s=70 , edgecolor='none',  marker='o', label='codebook_ok')
				pylab.scatter(W[i_ok , 0], W[i_ok,1],    color=colors[i_ok],   s=activations_perc[i_ok].T , edgecolor='none',  marker='+')
				pylab.scatter(W[i_ko ,0],  W[i_ko,1],    color='m',            s=50 , edgecolor='none',  marker='x', label='codebook_anomaly')
				for count, i in enumerate(W):
					pylab.annotate(count, xy = i[:2], xytext = (0, 0), textcoords = 'offset points')
				pylab.legend()

				if self.cluster_model is not None:
					pylab.figure('plot_Codebook_Cluster_Outlier' + fig_title)
					colors =  pylab.cm.Paired( np.linspace(0.1, 0.9,  self.num_units*2) )
					d2s=None
					if DEBUG_SOM:
						d2s= (data2saliency_index*5)**3+5
					i_ok = data2cluster>0
					i_ko = data2cluster<0
					pylab.scatter(X[i_ok,0], X[i_ok,1],  color=colors[data2cluster[i_ok]],  s=d2s, alpha=0.5)
					pylab.scatter(X[i_ko,0], X[i_ko,1],  color=colors[-data2cluster[i_ok]], s=d2s, alpha=0.5)
					pylab.scatter(X[data2saliency == False, 0], X[data2saliency == False, 1],  color='r',   s=40 , edgecolor='none', marker='x', label='data_anomaly')

					if labels:
						for label, x, y in zip(labels, list(X[:,0]), list(X[:,1])):
							try:
								pylab.annotate(label, (x, y), size=7)
							except:
								pass

					i_ok = unit2saliency == True
					i_ko = unit2saliency == False
					pylab.scatter(W[i_ok , 0], W[i_ok , 1],  color='k',   s=50 , edgecolor='none',  marker='o', label='codebook_ok')
					pylab.scatter(W[i_ko ,0],  W[i_ko,1],    color='m',   s=50 , edgecolor='none',  marker='x', label='codebook_anomaly')

					i_ok = centers2cluster>0
					i_ko = centers2cluster<0
					pylab.scatter(Centers[i_ok,0], Centers[i_ok,1],  color=colors[centers2cluster[i_ok]],   s=70 , edgecolor='k',  marker='v', label='CentersOfCodebook')
					pylab.scatter(Centers[i_ko,0], Centers[i_ko,1],  color=colors[-centers2cluster[i_ko]],  s=70 , edgecolor='m',  marker='v', label='CentersOfCodebook(Anomaly)')
					pylab.legend()
			
			else:
							
				fig=plt.figure('plot_Codebook_Outlier_3D' + fig_title)
				ax=fig.add_subplot(111, projection='3d')
				colors =  pylab.cm.Paired( np.linspace(0.1, 0.9,  self.num_units) )  #+  list( pylab.cm.gnuplot2( np.linspace(0.1, 0.9,  Nc) ) ) )[:self.num_units]
				# colors = pylab.cm.rainbow(np.linspace(0, 0.9, Nc))
				d2s=None
				if DEBUG_SOM:
					d2s= (data2saliency_index*5)**3+5
				ax.scatter(X[:,0], X[:,1],X[:,2],  color=colors[data2unit], s=40, alpha=0.5)
				ax.scatter(X[data2saliency == False, 0], X[data2saliency == False, 1], X[data2saliency == False, 2],  color='r',   s=40 , edgecolor='none', marker='x', label='data_anomaly')

				if labels:
					for label, x, y in zip(labels, list(X[:,0]), list(X[:,1]),list(X[:,2])):
						try:
							ax.annotate(label, (x, y), size=7)
						except:
							pass

				i_ok = unit2saliency == True
				i_ko = unit2saliency == False
				ax.scatter(W[i_ok , 0], W[i_ok , 1], W[i_ok , 2],  color='k',   s=100 , edgecolor='none',  marker='o', label='codebook_ok')
				ax.scatter(W[i_ok , 0], W[i_ok,1], W[i_ok , 2],    color=colors[i_ok],   s=activations_perc[i_ok].T , edgecolor='none',  marker='+')
				ax.scatter(W[i_ko ,0],  W[i_ko,1], W[i_ko,2],    color='m',            s=100 , edgecolor='none',  marker='x', label='codebook_anomaly')
				for count, i in enumerate(W):
					ax.annotate(count, xy = i[:2], xytext = (0, 0), textcoords = 'offset points')
				ax.legend()

				if self.cluster_model is not None:
					fig2=plt.figure('plot_Codebook_Cluster_Outlier' + fig_title)
					ax=fig2.add_subplot(111, projection='3d')
					colors =  plt.cm.Paired( np.linspace(0.1, 0.9,  self.num_units*2) )
					d2s=None
					if DEBUG_SOM:
						d2s= (data2saliency_index*5)**3+5
					i_ok = data2cluster>0
					i_ko = data2cluster<0
					ax.scatter(X[i_ok,0], X[i_ok,1],X[i_ok,2],  color=colors[data2cluster[i_ok]],  s=70, alpha=0.5)
					ax.scatter(X[i_ko,0], X[i_ko,1],X[i_ko,2],  color=colors[-data2cluster[i_ko]], s=70, alpha=0.5)
					ax.scatter(X[data2saliency == False, 0], X[data2saliency == False, 1], X[data2saliency == False, 2],  color='r',   s=40 , edgecolor='none', marker='x', label='data_anomaly')

					if labels:
						for label, x, y in zip(labels, list(X[:,0]), list(X[:,1]),list(X[:,2])):
							try:
								ax.annotate(label, (x, y), size=7)
							except:
								pass

					i_ok = unit2saliency == True
					i_ko = unit2saliency == False
					ax.scatter(W[i_ok , 0], W[i_ok , 1], W[i_ok , 2],  color='k',   s=100 , edgecolor='none',  marker='o', label='codebook_ok')
					ax.scatter(W[i_ko ,0], W[i_ko,1], W[i_ko , 2],    color='m',   s=100 , edgecolor='none',  marker='x', label='codebook_anomaly')

					i_ok = centers2cluster>0
					i_ko = centers2cluster<0
					ax.scatter(Centers[i_ok,0], Centers[i_ok,1], Centers[i_ok,2],  color=colors[centers2cluster[i_ok]],   s=70 , edgecolor='k',  marker='v', label='CentersOfCodebook')
					ax.scatter(Centers[i_ko,0], Centers[i_ko,1], Centers[i_ko,2],  color=colors[-centers2cluster[i_ko]],  s=70 , edgecolor='m',  marker='v', label='CentersOfCodebook(Anomaly)')
					ax.legend()

        if plot_type in (1,2):
            fig = pylab.figure('plot_units_outlier(all_components)' + fig_title)
            pylab.plot(X.T, color='b')
            if len(X[data2saliency == False])>0:
                pylab.plot(X[data2saliency == False].T,   color='r', marker='o') #Outlier
            if len(W[unit2saliency == True])>0:
                pylab.plot(W[unit2saliency == True].T,    color='g', marker='+') #Units ok
            if len(W[unit2saliency == False])>0:
                pylab.plot(W[unit2saliency == False].T,   color='m', marker='+') #Units outlier

        return  fig


    def plot_codebook_weights(self, interpolation       =   'bicubic',
                                    cmap                =   'jet',#'seismic',
                                    title               =   '' ):
        # # WEIGHTS OF CODEBOOK
        pylab.figure()
        pylab.title('WEIGHTS OF CODEBOOKS '+title)
        labels = np.array([str(unit_id) for unit_id in range(self.num_units)])
        labels = np.array(['U=%s/C=%d'%(labels[k],self.unit2cluster[k])    for k in range(self.num_units)])
        i_ord = np.argsort(self.unit2cluster)
        pylab.imshow(self.W[i_ord], origin='lower',aspect='auto', interpolation=interpolation, cmap=cmap)
        ax = pylab.gca()
        ax.set_xticks(range(self.W.shape[1]))
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels[i_ord])
        pylab.ylabel('codebooks')
        pylab.xlabel('weights')

        pylab.colorbar()


    def plot_mapping(self,  X               =   None,
                            target          =   None,
                            interpolation   =   'bicubic',
                            cmap            =   'jet',#'seismic',
                            title           =   '',
                            use_colors      =  False,
                     ):


        #UMatrix:
        distance_map, unit_xy = self.evaluate_UMatrix()
        pylab.figure()
        pylab.bone()
        pylab.imshow(distance_map, origin='lower',aspect='auto', interpolation=interpolation, cmap=cmap)
        pylab.xticks(range(self.width))
        pylab.yticks(range(self.height))
        pylab.colorbar()
        if X is None:
            pylab.title('CODEBOOKS ON UMATRIX  '+title)
            for unit_id in range(self.num_units):
                i, j = self.unit2cell(unit_id)
                pylab.annotate(unit_xy[i,j], xy = (j,i), xytext = (0, 0), textcoords = 'offset points', color='k',  size=8)
        else:
            pylab.title('DATA ON UMATIX '+title)
            # if target is None:
            #     target = np.arange(X.shape[0])
            data2cell = self.predict(X)[1]
            # num_labels = len( np.unique(target) )
            # markers = ( 'o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X')
            # Nmarkers = len(markers)
            if target is not None:
                if use_colors:
                    le     = LabelEncoder().fit(target)
                    Nc = len(np.unique(target))/2 + 1
                    colors = list( pylab.cm.Paired( np.linspace(0., 1.,  Nc) ) )  +  list( pylab.cm.gnuplot2( np.linspace(0., 1.,  Nc) ) )
                for  cell, t in zip(data2cell, target):
                    # palce a marker on the winning position for the sample
                    # pylab.plot(cell[0], cell[1], markers[t%Nmarkers], markerfacecolor='None', markeredgecolor=pylab.cm.Dark2(t), markersize=10, markeredgewidth=2)
                    # pylab.text(cell[0]+t*0.01,  cell[1]+t*0.01,  str(t), color=pylab.cm.Dark2(t/4), fontdict={'weight': 'normal',  'size': 12})
                    # dt = t*0.01
                    # if dt>0.1:
                    #     dt=0.1
                    # i , j  =  cell[0]+dt,  cell[1]+dt
                    i , j  =  cell[0] ,  cell[1]
                    if use_colors:
                        pylab.text(i, j,  str(t), color=colors[le.transform([t])[0]],  ha='center', va='center', size=8)
                    else:
                        pylab.text(i, j,  str(t), color='k',  ha='center', va='center', size=8)

        print ''


    def plot_mapping_cluster(self,  X                       =   None,
                                    interpolation           =   'bicubic',
                                    cmap                    =   'jet',
                                    title                   =   '',
                                    use_colors              =   False,
                             ) :
        if X is None:
            X = self.W

        if self.cluster_model is None:
            pylog.warn('CLUSTER MODEL not trained!')
            return None

        data2cluster = self.predict_cluster(X)

        self.plot_mapping(  X               =  X,
                            target          =  data2cluster,
                            interpolation   =  interpolation,
                            cmap            =  cmap,
                            title           = '(CENTERS)'+title,
                            use_colors      =  use_colors)

        pylab.figure()
        pylab.title('WEIGHTS OF CENTERS '+title)
        Nc = len(np.unique(self.unit2cluster))/2 + 10
        colors = list( pylab.cm.Paired( np.linspace(0., 1.,  Nc) ) )  +  list( pylab.cm.gnuplot2( np.linspace(0., 1.,  Nc) ) )
        markers = ['+', 'x','<','>']
        for k in np.unique(self.unit2cluster):
            ck = self.W[self.unit2cluster==k].mean(axis=0)
            if k<0:
                pylab.plot(ck, '--%s'%markers[-k%4],label=str(k), color=colors[-k])
            else:
                pylab.plot(ck, '-%s'%markers[k%4],label=str(k), color=colors[k])
        pylab.legend(loc="upper left", bbox_to_anchor=[0, 1], ncol=2, shadow=True, title="Legend", fancybox=True)


    def cellFrequencyDistribution(self, X=None, percent=True):
        if X is None:
            X = self.X
        data2cell = self.predict(X)[1]
        CD = np.zeros((self.height, self.width), dtype=int)
        for i,j in data2cell:
            CD[i,j] += 1
        if percent  and  np.sum(CD)>0:
            CD = CD / float(np.sum(CD))
        return  CD


    def cellComponentDistribution(self, X=None, component=0):
        if X is None:
            X = self.X
        data2cell = self.predict(X)[1]
        CD = np.zeros((self.height, self.width), dtype=float)
        MD = np.zeros((self.height, self.width), dtype=float)
        for  k in range(X.shape[0]):
            imin, jmin  =  data2cell[k]
            CD[imin,jmin] += 1.0
            MD[imin,jmin] += X[k,component]
            # print imin,jmin,'--->',X[k,component]
        MD /= CD
        return  MD


    def  plot_cellDistribution( self,  Data            =   None,
                                       component       =   None,
                                       percent         =   True,
                                       interpolation   =   'bicubic',
                                       cmap            =   'jet',  #seismic
                                       title           =   '',
                                       annotate        =   False ):

        if Data is None  and  component is None:
            title = 'CODEBOOKS component_0 distribution ' + title
            component = 0
            Data      = self.W

        elif Data is None  and  component is not None:
            Data      =self.W
            title = 'CODEBOOKS component_%d distribution '%component + title

        elif Data is not None  and  component is None:
            title = 'DATA frequency distribution ' + title

        elif Data is not None  and  component is not None:
            title = 'DATA component_%d distribution '%component + title

        if component is None:
            CD = self.cellFrequencyDistribution(Data, percent)
        else:
            CD = self.cellComponentDistribution(Data, component)
        CD = CD.T

        pylab.matshow(np.nan_to_num(CD,np.nanmin(CD)-1000.), cmap=cmap, origin='lower',interpolation=interpolation)
        pylab.gca().xaxis.set_ticks_position('bottom')
        pylab.colorbar()

        for (i, j), z in np.ndenumerate(CD):
            if isnan(z):
                pylab.text(j, i, 'x', ha='center', va='center', size=10)
            elif annotate:
                pylab.text(j, i, '{:0.2f}'.format(z), ha='center', va='center', size=8)

        pylab.title(title)



#################################################################################################################################################################################
##############################################################################DEMO###############################################################################################
#################################################################################################################################################################################



def create3Cluster(data_size=(10, 10, 10), array_size=20, sigma=.2, make_figure=True, title='3DataCluster'):
    from sklearn.utils import shuffle
    c1 = np.ones(array_size)*0.0  +  np.random.randn(data_size[0],  array_size)*sigma
    c2 = np.ones(array_size)*1.0  +  np.random.randn(data_size[1],  array_size)*sigma
    c3 = np.ones(array_size)*2.0  +  np.random.randn(data_size[2], array_size)*sigma

    Data    =  np.float32(np.concatenate((c1, c2, c3)))
    colors  =  np.array( ["blue"]*data_size[0] + ["red"]*data_size[1] + ["green"]*data_size[2] )
    labels  =  np.array( [0]*data_size[0] + [1]*data_size[1] + [2]*data_size[2] )
    # labels = range(len(colors))
    Data[0:int(data_size[0]*0.15),0] *=  10
    if make_figure:
        pylab.figure(title)
        pylab.scatter(Data[:, 0], Data[:, 1], c=colors)
        for k in range(len(labels)):
            pylab.annotate(labels[k], (Data[k,0], Data[k,1]), size=10)

    ix = shuffle(np.arange(Data.shape[0]))
    return Data[ix], colors[ix], labels[ix]



def prova_base():
    N     =  5
    sigma =  0.1
    som   =  ASOM(alpha_max=0.2, alpha_min=0.01, height=3, width=3, outlier_unit_threshold=0.1, outlier_percentile=98., Koutlier_percentile=2.)

    Data, colors, labels = create3Cluster(data_size=(5*N, 10*N, 15*N), sigma=sigma, array_size=2)
    som.train_batch(Data, num_epoch=60, training_type='iterative', verbose=2)
    som.plot_units_clusters_outlier(model_features_reduction=None)
    pylab.show()


    #Automatic clustering with SOM:
    som.fit_cluster(cluster_model=None, perc_subsampling=20., default_cluster_model=1, make_figure=True)
    som.plot_activations()
    pylab.show()

    #Explicit clustering with SOM:
    cluster_model = KMeans(n_clusters=3)
    som.fit_cluster(cluster_model=cluster_model, default_cluster_model=0, make_figure=True)
    som.plot_units_clusters_outlier(model_features_reduction=None)
    pylab.show()

    #plot:
    som.plot_units_clusters_outlier(model_features_reduction=None)
    som.plot_units_clusters_outlier(model_features_reduction=None)
    pylab.show()

    som.plot_codebook_weights()

    som.plot_mapping()
    som.plot_mapping(Data, target=labels, title='XXXXXXXXXXXXXXXXXX')
    som.plot_mapping_cluster()
    som.cellFrequencyDistribution()
    som.plot_cellDistribution(component=0, annotate=True)

    pylab.show()


    Data, colors, labels = create3Cluster(data_size=(10 * N, 0, 0), sigma=sigma)
    som.plot_cellDistribution(Data, title='Cluster0',component=0)

    Data, colors, labels = create3Cluster(data_size=(0, 10 * N, 0), sigma=sigma)
    som.plot_cellDistribution(Data, title='Cluster1', component=0)

    Data, colors, labels = create3Cluster(data_size=(0, 0, 10 * N), sigma=sigma)
    som.plot_cellDistribution(Data, title='Cluster2', component=0)




def prova_train_adaptive():
    from sklearn.datasets.samples_generator import make_blobs

    N          = 5
    sigma      = .1
    som = ASOM(alpha_max=0.2, alpha_min=0.01, height=1, width=3, outlier_unit_threshold=0.1, outlier_percentile=98., Koutlier_percentile=2.,
               initialize_unit_with_cluster=2, initialize_unit_with_cluster_quantile=0.01)

    #Test training online:
    Niter=100
    # step = int(Data.shape[0]/Niter)
    for k in range(1,Niter):
        centers = [[k*0.5+i, k*0.5+i] for i in range(4)]
        print centers
        X, _ = make_blobs(n_samples=600, centers=centers, cluster_std=0.1)
        X[0:3]=20
        if k==2:
            X[:10]=10
        if k==5:
            centers = [[30+i, 30+i] for i in range(6)]
            X, _ = make_blobs(n_samples=600, centers=centers, cluster_std=0.1)
        print "kkkkkkkkkkk=",k
        som.train_batch(X, num_epoch=50, training_type='iterative', verbose=2)
        #plot:
        som.plot_units_clusters_outlier(model_features_reduction=None)
        som.plot_activations()
        # som.plot_cellDistribution()
        # som.plot_cellDistribution(Data)
        # som.plot_cellDistribution(component=0)
        # som.plot_cellDistribution(component=1)
        # som.plot_cellDistribution(component=0)
        # som.plot_mapping()
        # som.plot_mapping(X, target=y)
        # som.fit_cluster(cluster_model=KMeans(n_clusters=4))
        # som.plot_mapping_cluster(X)
        # som.plot_codebook_weights()
        pylab.show()
        print

    # TEST_TRAIN_ONLINE=True
    # if TEST_TRAIN_ONLINE:
    #     for k in range(30):
    #         sigma=.1*k
    #         Data, colors, labels = create3Cluster(data_size=(15, 5, 5), sigma=sigma, make_figure=True)
    #         som.train_batch(Data, num_epoch=100, verbose=2)
    #         som.plot_mapping(Data, target=labels)
    #         som.plot_cellDistribution(Data,component=0)
    #         som.plot_units_clusters_outlier(model_features_reduction=None)
    #         som.plot_mapping_cluster(Data, cluster_model=cluster_model)
    #
    #         pylab.show()
    # som.plot_codebook_weights(weights_on_xaxis=True)
    # som.plot_codebook_weights(weights_on_xaxis=False)

    # som.train_batch(Data, num_epoch=60, verbose=1)
    # som.plot_mapping(Data, target=labels)
    # som.plot_mapping_cluster(cluster_model=MiniBatchKMeans(n_clusters=4))
    # som.plot_cellDistribution( component=0, annotate=True)

    # pylab.show()
    # som.plot_mapping(X=Data, target=labels)
    som.plot_units_clusters_outlier(model_features_reduction=None)
    # som.plot_mapping_cluster( )
    # som.plot_mapping_cluster(cluster_model=MiniBatchKMeans(n_clusters=4))


def prova_automatic_clustering():
    from sklearn.datasets.samples_generator import make_blobs
    from utilityAdvancedSOM.serializator import save_obj, load_obj

    som = ASOM(alpha_max=0.2, alpha_min=0.01, height=3, width=3, outlier_unit_threshold=0.1, outlier_percentile=98., Koutlier_percentile=2.)

    centers = [[k, k] for k in range(4)]
    X, _ = make_blobs(n_samples=800, centers=centers, cluster_std=0.1)
    Na=10
    X[:Na]=5 + np.random.randn(Na,2)
    X[:2]=  -1 + 0.1*np.random.randn(2,2)
    # X[-20:]=5*np.random.randn(20,2)

    # def read_in_chunks(file_object, chunk_size=2*30, n_skip=5):
    #     for k in range(n_skip):
    #         head = file_object.readline()
    #     while True:
    #         data = file_object.readlines(chunk_size)
    #         if not data:
    #             break
    #         yield np.genfromtxt(data, dtype=float, delimiter=',')
    #
    # infile = open("./test/data/FroudEnel.csv", "r")
    # for X in read_in_chunks(infile, 10):
    #     print X[:,0]
    #     raw_input()
    # som.train_batch(X, num_epoch=400, training_type = 'iterative',verbose=1, )
    if False:
        X = np.loadtxt("./test/data/EE_ASOM.csv", delimiter=',', skiprows=1)

        som.train_batch(X, num_epoch=50, training_type = 'adaptive',verbose=2, batch_size=1000)

        save_obj(som,"./test/data/EE_ASOM.bin")

    if False:
        som = load_obj("./test/data/EE_ASOM.bin")
    else:
        som.train_batch(X, num_epoch=50, training_type = 'adaptive',verbose=2, batch_size=100)

    #plot:
    som.fit_cluster(cluster_model=None,
                    perc_subsampling=5.,
                    default_cluster_model=0,
                    num_cluster_min=2,
                    num_cluster_max=20,
                    make_figure=1, )
    som.plot_mapping_cluster( )
    som.plot_cellDistribution()
    som.plot_units_clusters_outlier(model_features_reduction=None)#model_features_reduction=PCA(n_components=2))
    som.plot_activations()
    print som.cluster_model_centers
    print len(som.cluster_model_centers)

    hdb = HDBSCAN4ASOM(min_cluster_size=30, min_samples=20, prediction_data=True, gen_min_span_tree=True)
    Xsom = PCA(2).fit_transform(som.Xsom)
    hdb.fit(Xsom)
    pylab.figure()
    labels = som.cluster_model.labels_
    colors = pylab.cm.Paired( np.linspace(0., 1.,  len(np.unique(labels)) ) )
    c =  np.array([colors[l] if l >= 0 else (0., 0., 0.)   for l in labels])
    hdb.minimum_spanning_tree_.plot(edge_cmap='viridis',
                                        edge_alpha=0.6,
                                        node_size=80,
                                        edge_linewidth=2,
                                        node_color = c )



if __name__ == "__main__":
    # prova_base()
    prova_train_adaptive()
    # prova_automatic_clustering()
    pylab.show()

