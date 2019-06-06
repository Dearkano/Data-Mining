import numpy as np

import knn
import show_image
import extract_image

def hack(x_test):
    '''
    HACK Recognize a CAPTCHA image
      Inputs:
          img_name: filename of image
      Outputs:
          digits: 1x5 matrix, 5 digits in the input CAPTCHA image.
    '''
    data = np.load('hack_data.npz')
    print(data['arr_0'])

    # YOUR CODE HERE (you can delete the following code as you wish)
    x_train = data['arr_0']
    y_train = data['arr_1']
    digits = knn.knn(x_test,x_train,y_train,10)
    # begin answer
    # end answer

    return digits
