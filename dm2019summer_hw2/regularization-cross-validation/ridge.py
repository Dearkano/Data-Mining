import numpy as np

def ridge(X, y, lmbda):
    '''
    RIDGE Ridge Regression.

      INPUT:  X: training sample features, P-by-N matrix.
              y: training sample labels, 1-by-N row vector.
              lmbda: regularization parameter.

      OUTPUT: w: learned parameters, (P+1)-by-1 column vector.

    NOTE: You can use pinv() if the matrix is singular.
    '''
    P, N = X.shape
    w = np.zeros((P, 1))
    x = np.array(X)
    # YOUR CODE HERE
    # begin answer
    p1 = np.dot(x, x.T)
    p2 = lmbda * np.eye(P,P) 
    p3 = np.linalg.pinv(p1 + p2)
    p4 = np.dot(p3, x) 
    w = np.dot(p4, np.array([y]).T)
    # end answer
    return w
