import numpy as np

def perceptron(X, y):
    '''
    PERCEPTRON Perceptron Learning Algorithm.

       INPUT:  X: training sample features, P-by-N matrix.
               y: training sample labels, 1-by-N row vector.

       OUTPUT: w:    learned perceptron parameters, (P+1)-by-1 column vector.
               iter: number of iterations

    '''
    P, N = X.shape
    w = np.zeros((P + 1, 1))
    # YOUR CODE HERE
    # begin answer
    iters = 0
    b = np.ones((1, N))
    x = np.vstack((b, X))
    l = y[0]
    flag = True
    while flag and iters <= 10000:
        iters = iters + 1
        flag = False
        for i in range(N):
            a = np.array([x[:, i]])
            f = np.matmul(w.T, a.T)
            g = np.dot(f, l[i])
            if g <= 0:
                w = w + a.T * l[i]
                flag = True
    # end answer
    
    return w, iters