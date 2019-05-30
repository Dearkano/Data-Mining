import numpy as np

def sigmoid(inX):
    return (1 + np.exp(inX))

def logistic_r(X, y, lmbda):
    '''
    LR Logistic Regression.

    INPUT:  X: training sample features, P-by-N matrix.
            y: training sample labels, 1-by-N row vector.

    OUTPUT: w: learned parameters, (P+1)-by-1 column vector.
    '''
    P, N = X.shape
    w = np.ones((P, 1))
    r = 0.0001
    l = np.matrix(y)
    step = 0.001
    iter = 1000
    # YOUR CODE HERE
    # begin answer
    for i in range(iter):
        prev = w
        w = w - r * (np.sum(-y * X / ( 1 + np.exp(y * np.matmul(w.T, X))), axis=1).reshape(-1, 1)+ 2*lmbda*w)
        if(np.linalg.norm(w-prev)<step):
            break
    # end answer
    
    return w
