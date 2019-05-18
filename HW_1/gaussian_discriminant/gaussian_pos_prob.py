import numpy as np
import math

def gaussian_pos_prob(X, Mu, Sigma, Phi):
    '''
    GAUSSIAN_POS_PROB Posterior probability of GDA.
    Compute the posterior probability of given N data points X
    using Gaussian Discriminant Analysis where the K gaussian distributions
    are specified by Mu, Sigma and Phi.
    Inputs:
        'X'     - M-by-N numpy array, N data points of dimension M.
        'Mu'    - M-by-K numpy array, mean of K Gaussian distributions.
        'Sigma' - M-by-M-by-K  numpy array (yes, a 3D matrix), variance matrix of
                  K Gaussian distributions.
        'Phi'   - 1-by-K  numpy array, prior of K Gaussian distributions.
    Outputs:
        'p'     - N-by-K  numpy array, posterior probability of N data points
                with in K Gaussian distribsubplots_adjustutions.
    ''' 
    N = X.shape[1]
    K = Phi.shape[0]
    p = np.zeros((N, K))
    print(N)
    #Your code HERE
    # begin answer
    for i in range(0, N):
        temp = np.zeros(K)
        for j in range(0, K):
            x = X[:, i]
            mu = x - Mu[:, j]
            p1 =  1/(2 * math.pi * math.sqrt(np.linalg.det(Sigma[:, :, j])))
            p2 = math.exp(-1/2 * np.matmul(np.matmul(mu,  np.matrix(Sigma[:, :, j]).I), mu.T))
            p5 = Phi[j]
            temp[j] = p1 * p2
        p7 = np.matmul(temp, Phi.T)
        for j in range(0, K):
            p[i, j] = temp[j] * p5 / p7
    # end answer
    print(p)
    return p
# import numpy as np

# def gaussian_pos_prob(X, Mu, Sigma, Phi):
#     '''
#     GAUSSIAN_POS_PROB Posterior probability of GDA.
#     Compute the posterior probability of given N data points X
#     using Gaussian Discriminant Analysis where the K gaussian distributions
#     are specified by Mu, Sigma and Phi.
#     Inputs:
#         'X'     - M-by-N numpy array, N data points of dimension M.
#         'Mu'    - M-by-K numpy array, mean of K Gaussian distributions.
#         'Sigma' - M-by-M-by-K  numpy array (yes, a 3D matrix), variance matrix of
#                   K Gaussian distributions.
#         'Phi'   - 1-by-K  numpy array, prior of K Gaussian distributions.
#     Outputs:
#         'p'     - N-by-K  numpy array, posterior probability of N data points
#                 with in K Gaussian distribsubplots_adjustutions.
#     ''' 
#     N = X.shape[1]
#     K = Phi.shape[0]
#     p = np.zeros((N, K))
#     #Your code HERE

#     # begin answer
#     import math
#     for xi in range(0, N):
#         lk = np.zeros(K)
#         for i in range(0, K):
#             xmu = X[:, xi] - Mu[:, i]
#             lk[i] = 1 / (2 * math.pi * math.sqrt(np.linalg.det(Sigma[:, :, i]))) * math.exp(-0.5 * np.matmul(np.matmul(xmu, np.matrix(Sigma[:, :, i]).I), xmu.T))
#         px = np.matmul(lk, Phi.T)
#         for i in range(0, K):
#             p[xi, i] = lk[i] * Phi[i] / px
#     # end answer
    
#     return p
    