
from AdvancedSOM import ASOM
from AdvancedSOM import GapStatisticsKopt
from sklearn.cluster import KMeans
import numpy as np
import pylab
import  pandas as pd



som = ASOM(alpha_max=0.1, alpha_min=1e-8, height=2, width=2, outlier_unit_threshold=0.1,
           outlier_percentile=90, Koutlier_percentile=2, learning_rate_percentile=0.1,
           memory_size=None
           )






def findNumCluster(data, num_units_max=80):
    units = []
    costs = []
    data_anomaly = []
    units_anomaly = []
    for k in range(0,num_units_max,3):
        som = ASOM(alpha_max=0.1, alpha_min=0.0, height=1, width=k + 1, outlier_unit_threshold=0.2,
                   outlier_percentile=90, Koutlier_percentile=1, learning_rate_percentile=0.1,
                   memory_size=None)
        units.append(k+1)
        som.train_batch(data, num_epoch=200, batch_size=None, verbose=2, training_type='adaptive', fast_training=False)
        data_anomaly.append( np.sum(np.logical_not(som.data2saliency) ) )
        units_anomaly.append( np.sum(np.logical_not(som.unit2saliency) ) )
        costs.append(som.cost)
    costs = np.array(costs)
    dcosts = costs[1:]-costs[:-1]
    pylab.figure('costs-dcosts/data_anomaly/units_anomaly(num_units)')
    pylab.subplot(311)
    pylab.plot(units,costs,'.-')
    pylab.plot(units[1:],dcosts,'.-')
    pylab.subplot(312)
    pylab.plot(units,data_anomaly,'.-')
    pylab.subplot(313)
    pylab.plot(units,units_anomaly,'.-')

    # pylab.plot(dcosts,'.-')
    # pylab.plot(ddcosts,'.-')
    pylab.show()




center=0
def buildData(N=200, Na=10, dcenter=.2, prova=1):
    global center
    print "center=",center

    if prova==0:
        c1 = (center, center)  + np.random.rand(N, 2)/3
        ca = (-2., -2.)  + np.random.rand(Na, 2)*5
        data = np.concatenate((c1, ca))
    elif prova==1:
        ca = (-1,  -1)  + np.random.rand(Na, 2)*0.1
        c1 = (center, center)  + np.random.rand(N, 2)/3
        c2 = (1., 1.)  + np.random.rand(N, 2)/3
        # c2 = (0.2, 0.2)  + np.random.rand(N, 2)/3
        c3 = (2., 2.)  + np.random.rand(N, 2)/3
        # c3 = (.4, .4)  + np.random.rand(N, 2)/3
        cb = (0,  0)  + np.random.rand(Na, 2)
        data = np.concatenate((ca, c1, c2, c3, cb))
    elif prova==2:
        t1 = np.linspace(0,30,N)
        y1 = np.sin(2*np.pi*0.1*t1)   + np.random.rand(len(t1))
        t2 = np.linspace(0,30,N*2)
        y2 = (2*center+2)*np.sin(2*np.pi*0.1*t2) + np.random.rand(len(t2))
        data = np.concatenate((y1, y2))
        pylab.figure()
        pylab.subplot(211)
        pylab.axhline(3, c='r')
        pylab.plot(data)
        pylab.axhline(-2, c='r')
        LAG = 50
        data = np.array([data[k:k+LAG]  for k in range(len(data)-LAG)])
        pylab.subplot(212)
        pylab.plot(data.T)
        # pylab.show()
    elif prova==3:
        dlen = 700
        tetha = np.random.uniform(low=0,high=2*np.pi,size=dlen)[:,np.newaxis]
        X1 = 3*np.cos(tetha)+ .22*np.random.rand(dlen,1) + center
        Y1 = 3*np.sin(tetha)+ .22*np.random.rand(dlen,1)
        Data1 = np.concatenate((X1,Y1),axis=1)
        X2 = 1*np.cos(tetha)+ .22*np.random.rand(dlen,1)
        Y2 = 1*np.sin(tetha)+ .22*np.random.rand(dlen,1)
        Data2 = np.concatenate((X2,Y2),axis=1)
        X3 = 5*np.cos(tetha)+ .22*np.random.rand(dlen,1)
        Y3 = 5*np.sin(tetha)+ .22*np.random.rand(dlen,1)
        Data3 = np.concatenate((X3,Y3),axis=1)
        X4 = 8*np.cos(tetha)+ .22*np.random.rand(dlen,1)
        Y4 = 8*np.sin(tetha)+ .22*np.random.rand(dlen,1)
        Data4 = np.concatenate((X4,Y4),axis=1)
        data = np.concatenate((Data1,Data2,Data3,Data4),axis=0)

    elif prova==4:
        dlen = 200
        Data1 = pd.DataFrame(data= 1*np.random.rand(dlen,2))
        Data1.values[:,1] = (Data1.values[:,0][:,np.newaxis] + .42*np.random.rand(dlen,1))[:,0]
        Data2 = pd.DataFrame(data= 1*np.random.rand(dlen,2)+1)
        Data2.values[:,1] = (-1*Data2.values[:,0][:,np.newaxis] + .62*np.random.rand(dlen,1))[:,0]
        Data3 = pd.DataFrame(data= 1*np.random.rand(dlen,2)+2)
        Data3.values[:,1] = ((.5+center)*Data3.values[:,0][:,np.newaxis] + 1*np.random.rand(dlen,1))[:,0]
        Data4 = pd.DataFrame(data= 1*np.random.rand(dlen,2)+3.5)
        Data4.values[:,1] = (-.1*Data4.values[:,0][:,np.newaxis] + .5*np.random.rand(dlen,1))[:,0]
        data = np.concatenate((Data1,Data2,Data3,Data4))

    elif prova==5:
        data = np.load('./data/X.bin')
        data[-1,:] *= 2
        pylab.plot(data.T)
        pylab.show()

    if prova==6:

        x = np.linspace(0,10,100)
        y  = x*0
        c1 = np.array([x,y]).T

        x = np.linspace(0,5,100)
        y  = x
        c2 = np.array([x,y]).T

        x = np.linspace(5,10,100)
        y  = 10-x
        c3 = np.array([x,y]).T

        data = np.concatenate((c1, c2, c3))
        data += np.random.randn(*data.shape)*0.1

    if prova==7:

        x = np.linspace(0,10,100)
        y  = x*0
        c1 = np.array([x,y]).T

        x = np.linspace(0,10,100)
        y  = x*0+10
        c2 = np.array([x,y]).T

        y = np.linspace(0,10,100)
        x  = y*0
        c3 = np.array([x,y]).T

        y = np.linspace(0,10,100)
        x  = y*0 + 10
        c4 = np.array([x,y]).T

        y = np.linspace(0,10,100)
        x  = y
        c5 = np.array([x,y]).T

        data = np.concatenate((c1, c2, c3,c4,c5))
        data += np.random.randn(*data.shape)*0.1

    fig = pylab.figure()
    pylab.plot(data[:,0],data[:,1],'ob',alpha=0.2, markersize=4)
    pylab.show()


    center += dcenter

    return data



Outliers = []
def start(N=200, Na=30, num_epoch=300, alpha_max=0.1, train=True, training_type='adaptive', make_figure=True):


    data  =  buildData(N,Na, dcenter=0)

    # findNumCluster(data)

    print "-------------------TRAINING-------------------"
    som.train_batch(data, num_epoch=100, alpha_max=alpha_max, batch_size=None,verbose=2, training_type='onehot', fast_training=False)
    # som.train_batch_theano(data, num_epoch=num_epoch, alpha_max=alpha_max, batch_size=None,verbose=2, fast_training=False)
    # som.train_online(data, num_epoch=num_epoch, alpha_max=alpha_max, verbose=2, fast_training=True)

    clusters = som.data2unit #lista di appartenenza delle istances  (xk in X) ai relativi som units (uh in som_unit)
    print "clusters=",clusters
    print "Units=",som.W
    distances = som.data2dist  #lista delle distanze   delle istances (xk in X) ai relativi som units (uh in som_unit:  dist(xk,uh))
    # print "distances=",distances
    data2unit, data2cell, data2dist, data2saliency, data2saliency_prob, data2maps = som.predict(data)
    # print "som.unit2saliency=",som.unit2saliency #lista di codebook normali (salient unit)
    # print "som.data_saliency=",som.data_saliency #lista di data normali (salient instances)
    print "------som.X.shape  = ", som.X.shape
    print "--------------cost = ", som.cost
    o = np.sum(data2saliency == False)
    print "--------------Anomaly     =",o
    print "--------------Anomaly_perc=",o/float(data.shape[0])
    Outliers.append(o)

    print "TRANSFORM(X)=",som.transform(data, make_figure=True)
    # res = som.predict(som.W[som.unit2saliency])
    # W = res[5]
    # Wok = W[res[3]]
    # # Wok=som.W
    # gapS = GapStatisticsKopt(K=range(1, min(11, Wok.shape[0])))
    # kopt = gapS.findKopt(Wok)
    # print "gapStatisticsKMeans: optimal_k = ", kopt
    # clusters,data2cluster = gapS.getClusters(kopt,Wok)

    if True:
        # data2cluster = som.fit_predict_cluster(data, cluster_model=KMeans(n_clusters=8), make_figure=True)
        data2cluster = som.predict_cluster(data,  make_figure=True)
        # print data2cluster
        # width, height=som.evaluate_num_units_opt()
        # print "width, height=", width, height

    # for k in range(som.W.shape[0]):
    #     print "k_means:  k=%d     => labels=%d" %(k,k_means.labels_[k])


    if make_figure:
        # som.plot_clusters(data)
        # som.plot_units( )
        # som.plot_units_outlier( )
        som.plot_units_clusters_outlier(data, plot_type=2, model_features_reduction=None)
        # som.plot_activations()
        # som.plot_mapping()
        pylab.show()
        print


N  = 500
Na = 50
num_epoch=300
alpha_max = 0.1
num_call=0
training_step=1
train=True
for k in range(1000):
    start(N=N, Na=Na, num_epoch=num_epoch, alpha_max=alpha_max, train=train, make_figure=True)

    if False:
        num_epoch -= 50
        if num_epoch<=2:
            num_epoch=2
        print "num_epoch=",num_epoch

    if False:
        alpha_max -= 0.01
        if alpha_max<=1e-10:
            alpha_max=1e-10
        print "alpha_max=",alpha_max

    if False:
        train = False
        if num_call%training_step==0:
            train = True
        num_call+=1

    if True:
        N   += 1
        Na  += 20


    # pylab.figure('outliers')
    # pylab.plot(Outliers)
    # pylab.show()


##############################################################################DEMO###############################################################################################




if __name__ == "__main__":
    pylab.figure('outliers')
    pylab.plot(Outliers)
    start(N,Na, make_figure=True)
    pylab.show()

