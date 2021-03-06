import numpy as np
from scipy.optimize import minimize

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

    # YOUR CODE HERE
    # Please implement SVM with scipy.optimize. You should be able to implement
    # it within 20 lines of code. The optimization should converge wtih any method
    # that support constrain.
    # begin answer
    def getFun(py, px):
        return lambda W: py * np.dot(W, px) - 1
    f = lambda W: 1/2*np.dot(W[1: ], W[1: ])
    c = np.array([]).tolist()

    for i in range(N):
        c.append({
            'type': 'ineq',
            'fun': getFun(y[0, i], X[:, i])
        })
    res = minimize(f, X[:, 0], constraints=c)
    w = res.x
    for i in range(N):
        if(y[0, i] * np.dot(w, X[:, i])<=1.0001):
            num = num + 1
    w = np.array(w).reshape(-1, 1)
    # end answer
    return w, num

