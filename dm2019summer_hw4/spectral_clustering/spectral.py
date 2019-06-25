import numpy as np
from kmeans import kmeans 

def spectral(W, k):
    '''
    SPECTRUAL spectral clustering

        Input:
            W: Adjacency matrix, N-by-N matrix
            k: number of clusters

        Output:
            idx: data point cluster labels, n-by-1 vector.
    '''
    # YOUR CODE HERE
    # begin answer
    N = W.shape[0]
    D = (np.array(np.sum(W, axis=1)).T)[0]
    #D = np.array(np.sum(W, axis=1))
    L = np.diag(D) - W
    # DLD
    D_ = np.diag(1.0/np.sqrt(D))
    L = np.dot(np.dot(D_, L), D_)
    value, vector = np.linalg.eig(L)
    print(vector.shape)
    value = zip(value, range(N))
    value = sorted(value, key=lambda x:x[0])
    a, b = value[1]
    #H = vector[:, 1]
    H = (np.array(vector[:, b]).T)[0]
    t1 = np.mean(H)
    t2 = np.std(H)
    H = (H - t1) / t2
    H = np.array([H]).T
    res = kmeans(H, 2)
    return res
    # end answer
