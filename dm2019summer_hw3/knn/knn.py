import numpy as np
import math
import scipy.stats


def knn(x, x_train, y_train, k):
    '''
    KNN k-Nearest Neighbors Algorithm.

        INPUT:  x:         testing sample features, (N_test, P) matrix.
                x_train:   training sample features, (N, P) matrix.
                y_train:   training sample labels, (N, ) column vector.
                k:         the k in k-Nearest Neighbors

        OUTPUT: y    : predicted labels, (N_test, ) column vector.
    '''

    # Warning: uint8 matrix multiply uint8 matrix may cause overflow, take care
    # Hint: You may find numpy.argsort & scipy.stats.mode helpful

    # YOUR CODE HERE

    # begin answer
    N_test, P = x.shape
    N, P = x_train.shape
    dist = np.zeros((N_test, N))
    y = np.zeros(N_test)
    for i in range(N_test):
        for j in range(N):
            D = 0
            for s in range(P):
                D += (int(x[i, s]) - int(x_train[j, s]))**2
            dist[i, j] = math.sqrt(D)

    for i in range(N_test):
        tmp = np.argsort(dist[i])
        C = np.zeros(P)
        for j in range(k):
            C[int(y_train[tmp[j]])] += 1
        y[i] = np.argmax(C.astype(np.int))

    # end answer

    return y
