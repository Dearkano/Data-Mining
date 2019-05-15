import numpy as np

def likelihood(x):
    '''
    LIKELIHOOD Different Class Feature Liklihood 
    INPUT:  x, features of different class, C-By-N numpy array
            C is the number of classes, N is the number of different feature

    OUTPUT: l,  likelihood of each feature(from smallest feature to biggest feature) given by each class, C-By-N numpy array
    '''
    C, N = x.shape
    l = np.zeros((C, N))
    t = np.sum(x, axis=1)
    print(t)
    r = 0
    while r < C:
        m=0
        while m < N:
            l[r][m] = x[r][m] / t[r]
            m+=1
        r = r + 1 
    return l