import numpy as np
from likelihood import likelihood

def posterior(x):
    '''
    POSTERIOR Two Class Posterior Using Bayes Formula
    INPUT:  x, features of different class, C-By-N vector
            C is the number of classes, N is the number of different feature
    OUTPUT: p,  posterior of each class given by each feature, C-By-N matrix
    '''

    C, N = x.shape
    l = likelihood(x)
    print('----')
    print(x)
    print(l)
    total = np.sum(x, axis=1)
    r = 0


    while r < C:
        m=0
        while m < N:
            l[r][m] *= total[r]
            m +=1
        r+=1

    total = np.sum(l, axis=0)
    print(total)
    r = 0
    while r < C:
        m = 0
        while m < N:
            l[r][m] /= total[m]
            m += 1
        r += 1

    print(l)
    return l
