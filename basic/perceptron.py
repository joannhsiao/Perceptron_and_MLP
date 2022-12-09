import numpy as np
from sklearn.model_selection import train_test_split
from network import Dense
from activation import Sigmoid
from Loss import *

def predict(network, data):
    output = data
    for layer in network:
    	output = layer.forward(output)
    return output

def train(network, loss, loss_prime, x_train, y_train, epochs=50, learning_rate=0.01):
	predict_ans = []
	for epoch in range(epochs):
		error = 0
		acc = 0
		for x, y in zip(x_train, y_train):
			output = predict(network, x)
			error += loss(y, output)
			grad = loss_prime(y, output)	# backward
			for layer in reversed(network):
				grad = layer.backward(grad, learning_rate)
			acc += Acc(output, y)
			if epoch == epochs - 1:
				predict_ans.append(output)

		error /= len(x_train)
		acc /= len(x_train)
		
		print("[epoch {}] acc: {:g}, loss: {:g}".format(epoch, acc, error))
	return acc, predict_ans

def test(network, loss, x_test, y_test):
    error = 0
    acc = 0
    predict_ans = []
    for x, y in zip(x_test, y_test):
        output = predict(network, x)
        error += loss(y, output)
        acc += Acc(output, y)
        predict_ans.append(output)
    error /= len(x_test)
    acc /= len(x_test)

    print("Testing acc: {:g}, loss: {:g}".format(acc, error))
    return acc, predict_ans

def data_process(file):
	with open(file, "r") as f:
		lines = f.readlines()
	data = []
	y = []
	for num, line in enumerate(lines):
		xdata, ydata, ans = line.split()
		y.append(int(ans))
		data.append([float(xdata), float(ydata)])

	x_train, x_test, y_train, y_test = train_test_split(data, y, random_state=776, test_size=0.33)
	x_train = np.reshape(x_train, (len(x_train), 2, 1))
	x_test = np.reshape(x_test, (len(x_test), 2, 1))
	
	# normalization
	max_y = np.max(y_train)
	min_y = np.min(y_train)
	y_train = (y_train - min_y) / (max_y - min_y)
	y_test = (y_test - min_y) / (max_y - min_y)
	y_train = np.reshape(y_train, (len(y_train), 1, 1))
	y_test = np.reshape(y_test, (len(y_test), 1, 1))
	return x_train, x_test, y_train, y_test

def Acc(output, label):
	if abs(label - output) < 0.5:
		return 1
	else:
		return 0

def process(file, epoch, lr):
	network = [
		Dense(2, 1),	# one neuron
		Sigmoid()
	]
	x_train, x_test, y_train, y_test = data_process(file)
	Train_acc, train_predict = train(network, mse, mse_prime, x_train, y_train, epoch, lr)
	Test_acc, test_predict = test(network, mse, x_test, y_test)
	x_train = np.reshape(x_train, (len(x_train), 2))
	x_test = np.reshape(x_test, (len(x_test), 2))
	weights = network[len(network)-2].bias.ravel().tolist() + network[len(network)-2].weights.ravel().tolist()
	return Train_acc, x_train, y_train, Test_acc, x_test, y_test, weights, train_predict, test_predict