import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('data.txt')
target = np.loadtxt('label.txt')

split_point = int(data.shape[0]*0.75)

data_train = data[:split_point]
data_test = data[split_point:]

target_train = target[:split_point]
target_test = target[split_point:]


n = data_train.shape[1]
m = data_train.shape[0]
num_iters = 300
alpha = 0.01



theta = np.random.rand(n).T


target_hat = np.dot(data_train, theta)


cost = (1/(2*m))*np.sum((target_hat - target_train)**2)


cost_arr = []
iters = []

cost_arr.append(cost)
iters.append(0)


for i in range(num_iters):
	target_hat = np.dot(data_train, theta)
	theta = theta - alpha*(1/m)*np.dot(data_train.T, target_hat - target_train)
	target_hat = np.dot(data_train, theta)
	iters.append(i+1)
	cost = (1/(2*m))*np.sum((target_hat - target_train)**2)
	cost_arr.append(cost)


target_hat = np.dot(data_test, theta)
cost = (1/(2*m))*np.sum((target_hat - target_test)**2)

print(cost)

plt.scatter(iters, cost_arr);
plt.show();
