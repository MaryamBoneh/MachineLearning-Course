import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('linear_data_test.csv')
data = np.array(data)

X_train = np.array([data[:, 0], data[:, 1]])
Y_train = np.array(data[:, 2])

X_train = X_train.reshape(200, 2)

print(X_train)
print(Y_train)

N = 200
lr = 0.0001
epochs = 4
w = np.random.rand(2, 1)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

for i in range(epochs):
    for j in range(N):
        #train
        y_pred = np.matmul(X_train[j], w)
        e = Y_train[j] - y_pred
        w = w + e * lr * X_train[j]
        print(w, j)

        #plot
        ax.clear()
        X_plan, Y_plan = np.meshgrid(np.arange(X_train[:,0].min(), X_train[:,0].max()),
                                    np.arange(X_train[:,1].min(), X_train[:,1].max()))

        Z = X_plan * w[0] + Y_plan * w[1]

        mycmap = plt.get_cmap('gist_earth')
        ax.plot_surface(X_plan, Y_plan, Z, cmap = mycmap, alpha = 0.5)

        ax.scatter(X_train[:,0], X_train[:,1], Y_train, c= 'green')

        ax.set_xlabel('x')
        ax.set_ylabel('x')
        ax.set_zlabel('y')
        plt.pause(0.01)

plt.show()