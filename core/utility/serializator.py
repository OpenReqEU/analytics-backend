#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- coding: cp1252 -*-   #per non aver problemi con le vocali accentate.


import os
import cPickle

def save_obj(obj, filename="obj.bin", protocol=2):
    """
    Dumps obj Python to a file using cPickle.

    :Parameters:
        obj : object Python
        filename : str
            Path to the file where obj is dumped
    """
    if  not  os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    file = open(filename, 'wb')
    cPickle.dump(obj, file, protocol)
    file.close()
    return


def load_obj(filename="obj.bin"):
    """
    Loads obj Python pickled previously with `save_obj`.

    :Parameters:
        filename : str
            Path to the file with saved save_obj
    """
    file = open(filename, 'rb')
    obj = cPickle.load(file)
    return obj


def mytest(use_saved_obj=True):
    import  numpy
    import  time
    from    sklearn             import      cross_validation
    from    sklearn.metrics     import      mean_squared_error
    import  pylab
    from    sklearn.svm                 import  SVR

    import gc
    from monitorHW import  getMemoryUsagePerc

    # print "START)getMemoryUsagePerc  =", getMemoryUsagePerc()

    ###############################################################################
    # Load data
    # Create a the dataset
    rs          =   numpy.random.RandomState(13)
    n_samples   =   1500
    X0          =  rs.uniform(size=n_samples)*6
    #~ X0          =  numpy.sort(X0)
    X1          =  numpy.cos(X0)
    X2          =  numpy.exp(-X0)
    y           =  numpy.sin(X0).ravel() + numpy.sin(6 * X0).ravel() + rs.normal(0, 0.3, X0.shape[0])  +  X1.ravel() + X2.ravel()
    X           =  numpy.c_[X0, X0, X0, X2, X0, X0, X2, X2, X1, X2, X0]
    X           =  numpy.c_[X,X,X,X,X]
    X           =  numpy.c_[X,X,X,X,X]
    #~ X           =  numpy.c_[X0, X1, X2]

    print "X.shape=",X.shape
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3, random_state=9)
    if not use_saved_obj:
        dtr = SVR(gamma=0.1)

        dtr.fit(X_train, y_train)
        t=time.time()
        save_obj(dtr, "./xxx/SVM.obj")
        print "tempo save=",time.time()-t
        print "CREATO OBJ)getMemoryUsagePerc  =", getMemoryUsagePerc()
        del dtr
        gc.collect()

        print "DELETED OBJ)getMemoryUsagePerc  =", getMemoryUsagePerc()

    t=time.time()
    dtr = load_obj("./xxx/SVM.obj")
    print "tempo load=",time.time()-t
    print dtr
    print "LOADED OBJ)getMemoryUsagePerc  =", getMemoryUsagePerc()


    MSE_PS = mean_squared_error(y_test, dtr.predict(X_test))
    print("\n\nMSE_PS: %.4f" % MSE_PS)



    pylab.figure()
    pylab.plot(X_test[:,0], y_test,               ".b", label="y(PS)")
    pylab.plot(X_test[:,0], dtr.predict(X_test),  ".r", label="yp(PS)")
    pylab.legend(loc='upper right')

    del dtr
    gc.collect()
    print "DELETED OBJ)getMemoryUsagePerc  =", getMemoryUsagePerc()

    pylab.show()





def main():
    mytest(use_saved_obj=False)
    for k in range(10):
        mytest(use_saved_obj=True)





if __name__ == '__main__':
    main()













