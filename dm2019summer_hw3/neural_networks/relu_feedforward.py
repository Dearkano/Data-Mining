import numpy as np

def relu_feedforward(in_):
    '''
    The feedward process of relu
      in_:
              in_	: the input, shape: any shape of matrix
      
      outputs:
              out : the output, shape: same as in
    '''
    # TODO

    # begin answer
    P, N = in_.shape
    out = np.zeros((P, N))
    for i in range(P):
        for j in range(N):
            if(in_[i, j] > 0):
                out[i, j] = in_[i, j]
    # end answer
    return out
