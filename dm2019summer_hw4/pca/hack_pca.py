import numpy as np
import matplotlib.pyplot as plt
from pca import PCA 
import math
def meanX(dataX):
    return np.mean(dataX,axis=0)
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
   
def hack_pca(filename):
    '''
    Input: filename -- input image file name/path
    Output: img -- image without rotation
    '''
    img_r = (plt.imread(filename)).astype(np.float64)/255
    img_r = rgb2gray(img_r)
    plt.imshow(img_r, cmap='gray')
    plt.show()
    m, n = img_r.shape
    xy = []
    xyv = []
    for i in range(m):
        for j in range(n):
            if img_r[i, j] > 0 :
                xy.append((i, j)) 
                xyv.append((i, j, img_r[i, j]))
    xy = np.array(xy)
    vector, value = PCA(xy)
    d = np.array(np.round(np.dot(xy, vector))).astype(np.int)
    min_xy = np.min(d, axis=0)
    d -= min_xy
    max_xy = np.max(d, axis=0)
    img = np.zeros((max_xy[1]+1, max_xy[0]+1))
    for i in range(xy.shape[0]):
        img[max_xy[1] - d[i, 1],max_xy[0] - d[i, 0]] = xyv[i][2]
    plt.imshow(img, cmap='gray')
    plt.show()        
    return img
