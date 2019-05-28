import numpy as np
from cvxopt import solvers

def svm(X, y):
    '''
    SVM Support vector machine.

    INPUT:  X: training sample features, P-by-N matrix.
            y: training sample labels, 1-by-N row vector.

    OUTPUT: w: learned perceptron parameters, (P+1)-by-1 column vector.
            num: number of support vectors

    '''
    P, N = X.shape
    w = np.zeros((P, 1))
    num = 0
    b = np.ones((1, N))
    x = np.array(np.vstack((b, X)))
    d = np.matrix(x)

    # YOUR CODE HERE
    # Please implement SVM with scipy.optimize. You should be able to implement
    # it within 20 lines of code. The optimization should converge wtih any method
    # that support constrain.
    # begin answer
    G = -y * w.T
    q = 0
    p = np.eye(P)
    h = 1
    sol = solvers.qp(p,q,G,h)
    print(sol)
    # end answer
    return w, num

