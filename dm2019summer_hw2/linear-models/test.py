import numpy as np
from cvxopt import solvers
p = np.array([[1], [2], [3]])
k = np.array([[2,3,3],[4,5,6],[1,1,1]])
a = p * k
print(p[0:3])
