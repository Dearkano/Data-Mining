import numpy as np
from scipy.io import loadmat 

def likelihood(x):
    '''
    LIKELIHOOD Different Class Feature Liklihood 
    INPUT:  x, features of different class, C-By-N numpy array
            C is the number of classes, N is the number of different feature

    OUTPUT: l,  likelihood of each feature(from smallest feature to biggest feature) given by each class, C-By-N numpy array
    '''
    print ("hello")
    C, N = x.shape
    l = np.zeros((C, N))
    #TODO

    # begin answer
    # end answer

    return l

class A:
    pass
a = A()
a.shape = [1,2]
m = loadmat("./data.mat") 
print (m)