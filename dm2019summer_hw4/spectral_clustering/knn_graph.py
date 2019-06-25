import numpy as np
def euclidDistance(x1, x2):
    return np.sum((x1-x2)**2)

def calEuclidDistanceMatrix(X):
    X = np.array(X)
    S = np.zeros((len(X), len(X)))
    for i in range(len(X)):
        for j in range(i+1, len(X)):
            S[i][j] = 1.0 * euclidDistance(X[i], X[j])
            S[j][i] = S[i][j]
    return S

def knn_graph(X, k, threshold):
    '''
    KNN_GRAPH Construct W using KNN graph

        Input:
            X - data point features, n-by-p maxtirx.
            k - number of nn.
            threshold - distance threshold.

        Output:
            W - adjacency matrix, n-by-n matrix.
    '''

    # YOUR CODE HERE
    # begin answer
    N, P = X.shape
    A = np.zeros((N, N))
    S = calEuclidDistanceMatrix(X)
    # cal knn
    for i in range(N):
        dist_with_index = zip(S[i], range(N))
        dist_with_index = sorted(dist_with_index, key=lambda x:x[0])
        neighbours_id = [dist_with_index[m][1] for m in range(k+1)]

        for j in neighbours_id: 
            A[i][j] = np.exp(-S[i][j]/(2*threshold**2))
            A[j][i] = A[i][j] 

    return A
    # end answer
