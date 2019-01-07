#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- coding: cp1252 -*-

import random
import numpy as np



def buildKcluster2D(N, k, clear = True):
    '''
    this function generates some random samples with k clusters, the return array has two features/cols

    N: int, the number of datapoints
    k: int, the number of clusters
    '''
    n = float(N)/k
    X = []
    if clear:
        for i in range(k):
            c = (random.uniform(-2, 2), random.uniform(-2, 2))
            s = random.uniform(0.05,0.25)
            x = []
            while len(x) < n:
                a, b = np.array([np.random.normal(c[0], s), np.random.normal(c[1], s)])
                if abs(a) < 3 and abs(b) < 3:
                    x.append([a,b])
            X.extend(x)
    else:
        for i in range(k):
            c = (random.uniform(-1, 1), random.uniform(-1, 1))
            s = random.uniform(0.05,0.5)
            x = []
            while len(x) < n:
                a, b = np.array([np.random.normal(c[0], s), np.random.normal(c[1], s)])
                if abs(a) < 1 and abs(b) < 1:
                    x.append([a,b])
            X.extend(x)
    X = np.array(X)[:N]
    return X




def main():
    return 0


if __name__ == '__main__':
    main()
