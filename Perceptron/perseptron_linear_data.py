import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('csv_files/linear_data_train.csv')
test_data = pd.read_csv('csv_files/linear_data_test.csv')

data = np.array(data)
test_data = np.array(test_data)

X_train = np.array([data[:, 0], data[:, 1]])
Y_train = np.array(data[:, 2])
X_train = X_train.reshape(-1, 2)

X_test = np.array([data[:, 0], data[:, 1]])
Y_test = np.array(data[:, 2])
X_test = X_test.reshape(-1, 2)

print(X_train)
print(Y_train)

N = 1000
lr = 0.01
epochs = 4

w = np.random.rand(2, 1)

X_plan, Y_plan = np.meshgrid(np.arange(X_train[:,0].min(), X_train[:,0].max(), 1),
                            np.arange(X_train[:,1].min(), X_train[:,1].max(), 1))

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

for j in range(N):
    #train
    y_pred = np.matmul(X_train[j], w)
    e = Y_train[j] - y_pred
    w = w + e * lr * X_train[j]
    w = w.T
    print('w: ', w)

    #plot
    ax.clear()
    Z = Y_plan * w[0] +  X_plan * w[1]
    mycmap = plt.get_cmap('gist_earth')
    ax.plot_surface(X_plan, Y_plan, Z, cmap = mycmap, alpha = 0.5)
    ax.scatter(X_train[:,0], X_train[:,1], Y_train, c= 'green')

    ax.set_xlabel('x')
    ax.set_ylabel('x')
    ax.set_zlabel('y')
    plt.pause(0.01)

plt.show()


class Perceptron:
    def __init__(self):
        self.W = w

        
    def predict(self, X_test):
        Y_predic = np.matmul(X_test, self.W)
        return Y_predic
    

    def evaluation(self, X_test, Y_test):
        Y_predic = np.matmul(X_test, self.W)


        Y_predic[Y_predic > 0] = 1
        Y_predic[Y_predic<0] = -1             
        evaluation = np.count_nonzero(Y_predic != Y_test)
        
        for i in range(len(X_test)):
            print('X_test[i] : ', X_test[i])
            print('w: ', w)

        Y_predic = np.matmul(X_test, self.w)
        subtract = np.abs(Y_test - Y_predic)
        average = np.mean(subtract)

        mae=np.mean(subtract)
        mse=np.mean((subtract)**2)


        return average, mae, mse
model = Perceptron()
model.evaluation()