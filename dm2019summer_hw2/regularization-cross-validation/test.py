import numpy as np

a = np.array([[1,2,3,4],[2,3,4,5]])
print(np.hstack((a[:,0:3], a[:,4:4])))