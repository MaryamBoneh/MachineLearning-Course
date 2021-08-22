from os import error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('linear_data_train.csv')
data = np.array(data)

X_train = np.array([data[:, 0], data[:, 1]])
Y_train = np.array(data[:, 2])

X_train = X_train.reshape(1000, 2)

N = 1000
lr = 0.01
w = np.random.rand(2, 1)
lost = []

fig = plt.figure()
ax = fig.add_subplot()

for j in range(N):
    #train
    y_pred = np.matmul(X_train[j], w)
    e = Y_train[j] - y_pred
    w = w + e * lr * X_train[j]
    error = np.mean(w)
    lost.append(error)
    print('error: ', error)

    #plot
    ax.clear()
    ax.plot(lost)
    plt.pause(0.01)

plt.show()
