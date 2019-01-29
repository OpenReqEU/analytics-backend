#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 09:27:52 2018

@author: fra
"""
from __future__ import division

from sklearn.metrics import silhouette_score,calinski_harabaz_score
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA 
from scipy.spatial import distance
from sklearn.cluster import KMeans
import pandas as pd
import seaborn as sns
from collections import Counter
import numpy.matlib as matlib


def DaviesBouldin(X, labels):
    n_cluster = len(np.bincount(labels))
    cluster_k = [X[labels == k] for k in range(n_cluster)]
    centroids = [np.mean(k, axis = 0) for k in cluster_k]
    variances = [np.mean([distance.euclidean(p, centroids[i]) for p in k]) for i, k in enumerate(cluster_k)]
    db = []
    for i in range(n_cluster):
        for j in range(n_cluster):
            if j != i:
                db.append((variances[i] + variances[j]) / distance.euclidean(centroids[i], centroids[j]))
    return(np.max(db) / n_cluster)


def bench_k_means(km_fitted, data, sil_sample_size=300):
    km = km_fitted
    #elbow-method
    i = km.inertia_
    label = km.labels_
    s = silhouette_score(data, label, metric='euclidean',sample_size=sil_sample_size,random_state=0)
    ch = calinski_harabaz_score(data, label)
    db = DaviesBouldin(data, label)
    return s, i, ch, db


def get_knee(values):

    #get coordinates of all the points
    nPoints = len(values)
    allCoord = np.vstack((range(nPoints), values)).T
    #np.array([range(nPoints), values])
    
    # get the first point
    firstPoint = allCoord[0]
    # get vector between first and last point - this is the line
    lineVec = allCoord[-1] - allCoord[0]
    lineVecNorm = lineVec / np.sqrt(np.sum(lineVec**2))
    
    # find the distance from each point to the line: vector between all points and first point
    vecFromFirst = allCoord - firstPoint
    
    # To calculate the distance to the line, we split vecFromFirst into two omponents, one that is parallel to the line and one that is perpendicular 
    # Then, we take the norm of the part that is perpendicular to the line and get the distance.
    # We find the vector parallel to the line by projecting vecFromFirst onto the line. The perpendicular vector is vecFromFirst - vecFromFirstParallel
    # We project vecFromFirst by taking the scalar product of the vector with the unit vector that points in the direction of the line (this gives us 
    # the length of the projection of vecFromFirst onto the line). If we multiply the scalar product by the unit vector, we have vecFromFirstParallel
    scalarProduct = np.sum(vecFromFirst * matlib.repmat(lineVecNorm, nPoints, 1), axis=1)
    vecFromFirstParallel = np.outer(scalarProduct, lineVecNorm)
    vecToLine = vecFromFirst - vecFromFirstParallel
    
    # distance to line is the norm of vecToLine
    distToLine = np.sqrt(np.sum(vecToLine ** 2, axis=1))
    
    # knee/elbow is the point with max distance value
    idxOfBestPoint = np.argmax(distToLine)
    
    return values[idxOfBestPoint],idxOfBestPoint


def optimalK(k_cluster_range, data, verbose=True, getPlot=True):
    s=[];    i=[];    ch=[];    db=[]
    labels = []
    
    for k in k_cluster_range:
        if verbose: print('#### k:', k)
        km = KMeans(init='k-means++', n_clusters=k, n_init=10).fit(data)
        _labels = km.labels_
        _s, _i, _ch, _db = bench_k_means(km , data=data,sil_sample_size=500)
        if verbose: print('silhouette:',_s, 'inertia:', _i, 'calinski_harabaz:', _ch , 'davies_bouldin:', _db)
        s.append(_s)
        i.append(_i)
        ch.append(_ch)
        db.append(_db)
        labels.append(_labels)
        
    if getPlot:
        plot_metric(k_cluster_range,LinearScaling(s,0,1), color='b',yLabel='scaled metrics')
        plot_metric(k_cluster_range,LinearScaling(i,0,1), color='k',yLabel='scaled metrics')
        plot_metric(k_cluster_range,LinearScaling(ch,0,1), color='r',yLabel='scaled metrics')
        plot_metric(k_cluster_range,LinearScaling(db,0,1), color='g',yLabel='scaled metrics')
        plt.legend(['silhouette','inertia','calinski_harabaz','davies_bouldin'])
    
    s_knee = s[:] #make copy
    s_knee.remove(max(s_knee)) # usually max is 2
    s_knee = s.index(max(s_knee))
    _, i_knee = get_knee(i)
    _, ch_knee = get_knee(ch)
    _, db_knee = get_knee(db)
    
    k_s = k_cluster_range[s_knee]
    k_i = k_cluster_range[i_knee]
    k_ch = k_cluster_range[ch_knee]
    k_db = k_cluster_range[db_knee]
    best_k = (k_s + k_i + k_ch + k_db)/4.0
    
    res_dict = {'silhouette':{'bestK':k_s,'values':s},
                'inertia':{'bestK':k_i,'values':i},
                'calinski_harabaz':{'bestK':k_ch,'values':ch},
                'davies_bouldin':{'bestK':k_db,'values':db},
                'labels':labels}
    
    return best_k, res_dict


def plot_metric(cluster_range,metric, color='b',yLabel='metric'):
    plt.plot(cluster_range, metric, '-'+color+'o',mfc='none')  
    plt.xticks( cluster_range )
    plt.xlabel('cluster', fontsize=20)
    plt.ylabel(yLabel, fontsize=20)
    plt.tick_params(labelsize=15)
    plt.grid(color='#d3d3d3', linestyle='-', linewidth=1)


def plot_cluster_counts(labels, color='#6495ed',clustername='default', orient='v', order_by_count=False):
    sns.set(style="whitegrid")
    
    index = list(set(labels))
    
    if clustername !='default':
        replace_dict = dict([(k,v) for k,v in enumerate(clustername)])
        labels = list(labels)
        labels = [replace_dict[elem] for elem in labels ]
        index = clustername
    
    if order_by_count:
        lable_count = Counter(labels)
        index, _ = zip(*lable_count.most_common(len(lable_count)))
        index = list(index)
    if orient=='v':
        g = sns.countplot(x=labels, color=color, order=index  ) 
        g.set_ylabel('cluster',fontsize=20)
        g.set_ylabel('count elments in cluster',fontsize=20)
    else:
        g = sns.countplot(y=labels, color=color, order=index  )   
        g.set_ylabel('arguments',fontsize=20)
        g.set_xlabel('count docs by argument',fontsize=20)
    g.tick_params(labelsize=15)
    sns.plt.show()
    return g

def closest_to_centroids(data,centroids,nClosest):
    #from sklearn.metrics import pairwise_distances_argmin_min
    #closest, _ = pairwise_distances_argmin_min(centroids, mean_w2v_vec_list)
    kdtree = cKDTree(data)
    _, indices = kdtree.query(centroids, k=nClosest, p=2)
    return indices
    

def scatter_plot_2D(x,y,labels,centroids,label_centroids='default'):
    colmap = {1:'r', 2:'g', 3:'b', 4:'c', 5:'m', 6:'y', 7:'k' , 
              8:'orange' , 9:'purple', 10:'brown', 11:'pink', 12:'gray', 13:'olive'}
    colors = map(lambda x: colmap[x+1], labels)
    plt.scatter(x, y, color=colors, alpha=0.5, edgecolor='k')
    if label_centroids=='default':
        label_centroids = range(len(centroids))
    for idx, centroid in enumerate(centroids):
        plt.scatter(*centroid, color=colmap[idx+1], marker='^', s=100)
    plt.xlabel( 'first dimension' )
    plt.ylabel( 'second dimension' )


def scatter_plot_nD(data,labels,centroids,label_centroids='default', nRandSample='all'):
    if nRandSample != 'all':
        idx = np.random.randint(len(data), size=nRandSample)
        data = data[idx]
        labels = labels[idx]
    pca = PCA(n_components=2, svd_solver= 'randomized').fit(data)
    data = pca.transform(data) 
    centroids = pca.transform(centroids) 
    scatter_plot_2D(data[:,0],data[:,1],labels,centroids,label_centroids)  


def gap_statistic_method(data, nrefs=3, maxClusters=15):
    """
    Calculates KMeans optimal K using Gap Statistic from Tibshirani, Walther, Hastie
    Params:
        data: ndarry of shape (n_samples, n_features)
        nrefs: number of sample reference datasets to create
        maxClusters: Maximum number of clusters to test for
    Returns: (gaps, optimalK)
    """
    gaps = np.zeros((len(range(1, maxClusters)),))
    results = []
    for gap_index, k in enumerate(range(1, maxClusters)):

        # Holder for reference dispersion results
        refDisps = np.zeros(nrefs)

        # For n references, generate random sample and perform kmeans getting resulting dispersion of each loop
        for i in range(nrefs):
            
            # Create new random reference set
            randomReference = np.random.random_sample(size=data.shape)
            
            # Fit to it
            km = KMeans(k)
            km.fit(randomReference)
            
            refDisp = km.inertia_
            refDisps[i] = refDisp

        # Fit cluster to original data and create dispersion
        km = KMeans(k)
        km.fit(data)
        
        origDisp = km.inertia_

        # Calculate gap statistic
        gap = np.log(np.mean(refDisps)) - np.log(origDisp)

        # Assign this loop's gap statistic to gaps
        gaps[gap_index] = gap
        
        results.append(gap)

    return gaps.argmax() + 1, results  



def LinearScaling(var,newMin,newMax):
    # range of the old variable
    rangeVar=max(var)-min(var)
    # range of the new variable
    rangeNewVar=newMax-newMin
    # compute new variable
    scaled_var= [((x-min(var))*(rangeNewVar/rangeVar) + newMin) for x in var] 
    return scaled_var

   


# mapping function
def matchClasses(original_categories,assigned_clusters,verbose = False):
    
    mapTab = pd.crosstab(np.array(assigned_clusters),np.array(original_categories))
    catDict = {}
    matched_obs = 0

    labels = mapTab.columns.tolist()
    clusters = mapTab.index.tolist()
    tab_array = np.array(mapTab)
    
    for cl_id, cl in enumerate(clusters):
        max_obs = np.max(tab_array[cl_id])
        max_pos = int(np.where(tab_array[cl_id] == max_obs)[0])
        cat_pos = labels[max_pos]
        clu_tot = np.sum(tab_array[cl_id])
        clu_prec = np.round(max_obs/clu_tot*100,2)
        if verbose:
            print('cluster '+str(cl)+": "+cat_pos)
        catDict[cl] = {'cat':cat_pos,'obs':clu_tot,'match':max_obs,'prec':clu_prec}
        matched_obs += max_obs
    
    print('cases in matched pairs: '+str(np.round(matched_obs/len(original_categories)*100,2))+"%")
    return catDict







 
if __name__ == "__main__":
    
    dataset = np.array([
        [12, 20, 28, 18, 29, 33, 24, 45, 45, 52, 51, 52, 55, 53, 55, 61, 64, 69, 72],
        [39, 36, 30, 52, 54, 46, 55, 59, 63, 70, 66, 63, 58, 23, 14, 8, 19, 7, 24],
        [45, 38, 32, 65, 58, 49, 52, 55, 66, 75, 60, 68, 52, 20, 20, 10, 25, 5, 24],
        [40, 20, 41, 50, 50, 47, 64, 84, 45, 80, 80, 65, 60, 28, 24, 12, 14, 2, 30]
        ])
    
    ### 2-Dim
    data = np.transpose(dataset[0:2,:])            
    
    k_cluster_range = range(2,11)
    best_k, metrics_dict =  optimalK(k_cluster_range, data)
    
    print('best k for silhouette:',metrics_dict['silhouette']['bestK'], 
          'best k for inertia:', metrics_dict['inertia']['bestK'], 
          'best k for calinski_harabaz:', metrics_dict['calinski_harabaz']['bestK'] , 
          'best k for davies_bouldin:', metrics_dict['davies_bouldin']['bestK'])

    print('best k:',best_k)
    
    km = KMeans(n_clusters=int(best_k)).fit(data)

    labels = km.labels_
    centroids = km.cluster_centers_
    
    s, i, ch, db = bench_k_means(km, data)
    print('silhouette:',s, 'inertia:', i, 'calinski_harabaz:', ch , 'davies_bouldin:', db)
           
    scatter_plot_2D(data[:,0],data[:,1],labels,centroids)
    plt.close()

    
    ### n-Dim
    data = np.transpose(dataset)
    
    k_cluster_range = range(2,10)
    best_k, metrics_dict =  optimalK(k_cluster_range, data)
    
    print('best k for silhouette:',metrics_dict['silhouette']['bestK'], 
          'best k for inertia:', metrics_dict['inertia']['bestK'], 
          'best k for calinski_harabaz:', metrics_dict['calinski_harabaz']['bestK'] , 
          'best k for davies_bouldin:', metrics_dict['davies_bouldin']['bestK'])

    print('best k:',best_k)
    
    km = KMeans(n_clusters=int(best_k)).fit(data)
    labels = km.labels_
    centroids = km.cluster_centers_
    
    s, i, ch, db = bench_k_means(km, data)
    print('silhouette:',s, 'inertia:', i, 'calinski_harabaz:', ch , 'davies_bouldin:', db)
    plt.close()
    scatter_plot_nD(data,labels,centroids) 
    
    #from sklearn.cluster import AgglomerativeClustering
    # Affinity = {“euclidean”, “l1”, “l2”, “manhattan”,“cosine”}
    #### Linkage 
    #-Ward: minimizes the sum of squared differences within all clusters. It is a variance-minimizing approach and in this sense is similar to the k-means objective function but tackled with an agglomerative hierarchical approach.
    #-Maximum: or complete linkage minimizes the maximum distance between observations of pairs of clusters.
    #-Average: linkage minimizes the average of the distances between all observations of pairs of clusters.
    #Hclustering = AgglomerativeClustering(n_clusters=5,affinity='euclidean', linkage='ward')
    #Hclustering.fit(mean_w2v_vec_list)
    #label = Hclustering.labels_
