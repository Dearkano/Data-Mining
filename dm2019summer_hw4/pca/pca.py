import numpy as np
def meanX(dataX):
    return np.mean(dataX,axis=0, keepdims=True)
def PCA(data):
    '''
    PCA	Principal Component Analysis

    Input:
      data      - Data numpy array. Each row vector of fea is a data point.
    Output:
      eigvector - Each column is an embedding function, for a new
                  data point (row vector) x,  y = x*eigvector
                  will be the embedding result of x.
      eigvalue  - The sorted eigvalue of PCA eigen-problem.
    '''

    # YOUR CODE HERE
    # Hint: you may need to normalize the data before applying PCA
    # begin answer
    k = meanX(data)
    data = data - k
    data /= np.max(data)
    cov = np.cov(data.T)
    value, vector = np.linalg.eig(cov)
    N = len(value)
    x = zip(value, range(N))
    x = sorted(x, key=lambda x:x[0], reverse=True)
    vector = np.vstack([vector[:,i] for (v, i) in x[:N]]).T
    value = sorted(value,reverse=True)
    return vector, value
    # end answer