import numpy as np
import matplotlib.pyplot as plt

def data_Generator(N):
    X = np.random.uniform(0, 40, N)
    Y = (X * 2) + np.random.normal(0, 2, N)

    X = X.reshape(-1, 1)
    Y = Y.reshape(-1, 1)

    return X, Y

N = 200
lr = 0.0001

X_train, Y_train = data_Generator(N)

w = np.random.rand(1, 1)
print(w)

fig, ax = plt.subplots()

for i in range(N):
    #train
    y_pred = np.matmul(X_train[i], w)
    e = Y_train[i] - y_pred
    w = w + e * lr * X_train[i]
    print(w)

    #plot
    Y_pred = np.matmul(X_train, w)
    ax.clear()
    plt.scatter(X_train, Y_train, c='red')
    ax.plot(X_train, Y_pred, c='blue', lw= 2)
    plt.pause(0.01)