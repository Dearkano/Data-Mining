import numpy as np

def linear_regression(X, y):
    '''
    LINEAR_REGRESSION Linear Regression.

    INPUT:  X: training sample features, P-by-N matrix.
            y: training sample labels, 1-by-N row vector.

    OUTPUT: w: learned perceptron parameters, (P+1)-by-1 column vector.
    '''
    P, N = X.shape
    w = np.zeros((P + 1, 1))
    # YOUR CODE HERE
    # begin answer
    b = np.ones((1, N))
    x = np.array(np.vstack((b, X)))
    w = np.matmul(np.matmul((np.matrix(np.matmul(x, x.T))).I, x), np.matrix(y).T)
    # end answer
    return w
