# some basic imports
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

# read data
data = sio.loadmat('./data.mat')
x1_train, x1_test, x2_train, x2_test = data['x1_train'], data['x1_test'], data['x2_train'], data['x2_test']

all_x = np.concatenate([x1_train, x1_test, x2_train, x2_test], 1)
data_range = [np.min(all_x), np.max(all_x)]

from get_x_distribution import get_x_distribution

train_x = get_x_distribution(x1_train, x2_train, data_range)
test_x = get_x_distribution(x1_test, x2_test, data_range)
print('-===-=-=')
print(train_x)

from likelihood import likelihood
l = likelihood(train_x)
width = 0.35
p1 = plt.bar(np.arange(data_range[0], data_range[1] + 1), l.T[:,0], width)
p2 = plt.bar(np.arange(data_range[0], data_range[1] + 1) + width, l.T[:,1], width)
plt.xlabel('x')
plt.ylabel('$P(x|\omega)$')
plt.legend((p1[0], p2[0]), ('$\omega_1$', '$\omega_2$'))
plt.axis([data_range[0] - 1, data_range[1] + 1, 0, 0.5])
plt.show()

err = 0
C = l.shape[1]
i = 0

while(i < C):
    if l[0][i] < l[1][i]:
        err += test_x[0, i]
    elif l[0][i] > l[1][i]:
        err += test_x[1, i]
    i += 1
print(err)



from posterior import posterior
print('====')
print(train_x)
p = posterior(train_x)
width = 0.35
p1 = plt.bar(np.arange(data_range[0], data_range[1] + 1), p.T[:,0], width)
p2 = plt.bar(np.arange(data_range[0], data_range[1] + 1) + width, p.T[:,1], width)
plt.xlabel('x')
plt.ylabel('$P(\omega|x)$')
plt.legend((p1[0], p2[0]), ('$\omega_1$', '$\omega_2$'))
plt.axis([data_range[0] - 1, data_range[1] + 1, 0, 1.2])
plt.show()

err = 0
C = p.shape[1]
i = 0

while(i < C):
    if p[0][i] < p[1][i]:
        err += test_x[0, i]
    elif p[0][i] > p[1][i]:
        err += test_x[1, i]
    i += 1
print(err)

# total risk

i = 0
r11 = 0
r12 = 1
r21 = 2 
r22 = 0
total_risk = 0

while i < C :
    R1 = r11 * p[0][i] + r12 * p[1][i]
    R2 = r21 * p[0][i] + r22 * p[1][i]
    if R1 < R2:
        total_risk += R1
    else:
        total_risk += R2
    i += 1

print(total_risk)