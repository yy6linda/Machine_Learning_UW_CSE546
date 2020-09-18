import numpy as np
from matplotlib import pyplot as plt
n = 256
x = []
sigma = 1
for i in range(1,257):
    x.append(i/n)
x = np.array(x)
err = np.random.normal(0, sigma, n)
f = 4*np.sin(np.pi*x)*np.cos(6*np.pi*x**2)
y = 4*np.sin(np.pi*x)*np.cos(6*np.pi*x**2) + err

average_emperical_error = []
average_bias_squared = []
average_variance = []
average_error = []
m_list = [1,2,4,8,16,32]
for m in m_list:
    f_hat = []
    for j in range(0,int(n/m)):
        c_j = np.sum(y[j*m+1:(j+1)*m+1])/m
        #print(c_j)
        f_hat = f_hat + [c_j]*m
    #average imperical error
    aie = np.sum((f_hat-f)**2)/n
    average_emperical_error.append(aie)
    f_bar = []
    for j in range(0,int(n/m)):
        f_j = np.sum(f[j*m+1:(j+1)*m+1])/m
        f_bar = f_bar + [f_j]*m

    abias = np.sum((f_bar-f)**2)/n
    average_bias_squared.append(abias)
    avar = sigma**2/m
    average_variance.append(avar)
    average_error.append(abias + avar)

plt.plot(m_list, average_emperical_error, label = "average_imperical_error")
plt.plot(m_list, average_bias_squared, label = "average_bias_squared")
plt.plot(m_list, average_variance, label = "average_variance")
plt.plot(m_list, average_error, label = "average_error")
plt.xlabel('m')
plt.legend()
plt.savefig('b1.png')
