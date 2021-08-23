from os import error
import numpy as np
from numpy.core.fromnumeric import mean
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('Perceptron/linear_data_train.csv')
data = np.array(data)

X_train = np.array([data[:, 0], data[:, 1]])
Y_train = np.array(data[:, 2])

# print()
X_train = X_train.reshape(1000, 2)
Y_train = Y_train.reshape(1000, 1)


N = 1000
lr = 0.01
w = np.random.rand(2, 1)
loss = []

fig = plt.figure()
ax = fig.add_subplot()

for j in range(N):
    #train
    y_pred = np.matmul(X_train[j], w)
    e = Y_train[j] - y_pred
    w = w + e * lr * X_train[j]
    Y_pred = np.matmul(X_train, w)
    error = np.abs(np.mean(Y_train - Y_pred))
    loss.append(error)
    print('error: ', error)

    #plot
    ax.clear()
    ax.plot(loss)
    plt.pause(0.01)

plt.show()
