# some basic imports
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import math
from plot_ex1 import plot_ex1, figure

# mu: 2x1 matrix, e.g. np.array([[0,0]]).T
# Sigma: 2x2 matrix e.g. nnp.array([[1, 0], [0, 1]]).T
# phi: a number e.g 0.5

# change the value for specific decision boundary

mu0 = np.array([[0, 0]]).T
Sigma0 = np.array([[1, 0], [0, 1]]).T
mu1 = np.array([[0, 0]]).T
Sigma1 = np.array([[5, 0], [0, 5]]).T
phi = 0.5
# begin answer
# end answer

plot_ex1(mu0, Sigma0, mu1, Sigma1, phi, 'Line', 1)
figure