import numpy as np
import math
def distance(x1, x2):
    len = x1.shape[0]
    d = 0
    for i in range(len):
        d += x1[i]**2 + x2[i]**2
    return math.sqrt(d)
a = np.array([[1,3,5],[2,4,6]])
x1 = np.array([3, 4])
x2 = np.array([2,5])
print(distance(x1, x2))