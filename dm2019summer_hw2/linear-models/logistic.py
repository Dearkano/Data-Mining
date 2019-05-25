import numpy as np

def sigmoid(inX):
    return 1.0/(1 + np.exp(-inX))

def logistic(X, y):
    '''
    LR Logistic Regression.

    INPUT:  X: training sample features, P-by-N matrix.
            y: training sample labels, 1-by-N row vector.

    OUTPUT: w: learned parameters, (P+1)-by-1 column vector.
    '''
    P, N = X.shape
    w = np.array(np.ones((P + 1, 1)))
    r = 0.01
    l = np.matrix(y)
    b = np.ones((1, N))
    x = np.array(np.vstack((b, X)))
    d = np.matrix(x)
    iter = 100
    # YOUR CODE HERE
    # begin answer
    for i in range(iter):
        h = np.dot(d.T, w)
        loss = h - l.T
        g = np.dot(d, l.T) / (P + 1)
        w = w - r * g
    # end answer
    
    return w
