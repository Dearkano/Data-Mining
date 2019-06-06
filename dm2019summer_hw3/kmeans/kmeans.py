import numpy as np
import random
import math

def distance(x1, x2):
    len = x1.shape[0]
    d = 0
    for i in range(len):
        d += (x1[i] - x2[i])**2
    return math.sqrt(d)

def kmeans(x, k):
    '''
    KMEANS K-Means clustering algorithm

        Input:  x - data point features, n-by-p maxtirx.
                k - the number of clusters

        OUTPUT: idx  - cluster label
                ctrs - cluster centers, K-by-p matrix.
                iter_ctrs - cluster centers of each iteration, (iter, k, p)
                        3D matrix.
    '''
    # YOUR CODE HERE
    # begin answer
    N, P = x.shape
    max_iter = 30
    idx = np.array([random.randint(0, k) for _ in range(N)])
    ctrs = np.zeros((k, P))
    iter_ctrs = np.zeros((max_iter, k, P))
    centers = [random.randint(0, N-1) for _ in range(k)]
    for q in range(k):
        iter_ctrs[0, q] = x[centers[q]]
        ctrs[q] = x[centers[q]]
    for i in range(max_iter):
        print('iter={}'.format(i))
        cur_idx = np.zeros(N)
        cluster = np.zeros((k, N, P))
        for j in range(N):
            dist = np.zeros(k)
            p = x[j]
            for q in range(k):
                c = ctrs[q]
                dist[q] = distance(p, c)
            class_ = np.argmin(dist)
            cur_idx[j] = class_
            cluster[class_, j] = p

        cur_idx = cur_idx.astype(np.int)
        if (cur_idx == idx).all():
            print('break')
            break
        idx = cur_idx.astype(np.int)
        # update centers
        for j in range(k):
            samples = cluster[j]
            m = np.sum(samples, axis=0)/N
            iter_ctrs[i, j] = m
            ctrs[j] = m
    totalDist = 0
    for i in range(N):
        totalDist += distance(x[i], ctrs[idx[i]])
        
    # end answer
    return idx, ctrs, iter_ctrs, totalDist
