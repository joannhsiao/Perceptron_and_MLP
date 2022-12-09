import numpy as np
from sklearn.model_selection import train_test_split
from network import Dense
from activation import Tanh, Sigmoid
from Loss import *

def predict(network, data):
    output = data
    for layer in network:
    	output = layer.forward(output)
    return output

def train(network, loss, loss_prime, x_train, y_train, epochs=50, learning_rate=0.01):
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
		error /= len(x_train)
		acc /= len(x_train)
		
		print("[epoch {}] acc: {}, loss: {}".format(epoch, acc, error))
	return acc

def test(network, loss, x_test, y_test):
    error = 0
    acc = 0
    for x, y in zip(x_test, y_test):
        output = predict(network, x)
        error += loss(y, output)
        acc += Acc(output, y)
    error /= len(x_test)
    acc /= len(x_test)

    print("Testing acc: {:g}, loss: {:g}".format(acc, error))
    return acc

def data_process(file):
	with open(file, "r") as f:
		lines = f.readlines()
	data = []
	y = []
	for num, line in enumerate(lines):
		xdata = line.split()
		y.append(int(float(xdata[-1])))
		for i in range(len(xdata)):
			xdata[i] = float(xdata[i])
		data.append(xdata[0:-1])
		input_dim = len(xdata) - 1
	x_train, x_test, y_train, y_test = train_test_split(data, y, random_state=776, test_size=0.33)
	x_train = np.reshape(x_train, (len(x_train), input_dim, 1))
	x_test = np.reshape(x_test, (len(x_test), input_dim, 1))

	# normalization
	min_y = np.min(y_train)
	y_train -= min_y
	y_test -= min_y

	# one-hot embedding
	count_y = {}
	for i in range(len(y_train)):
		if y_train[i] not in count_y:
			count_y[y_train[i]] = 1
		else:
			continue
	classes = len(count_y)

	i = 0
	for x in count_y:
	    one_hot = [0] * classes
	    one_hot[i] = 1
	    count_y[x] = one_hot
	    i += 1
	y_train_one_hot = []
	for i in range(len(y_train)):
		y_train_one_hot.append(count_y[y_train[i]])
	y_test_one_hot = []
	for i in range(len(y_test)):
		y_test_one_hot.append(count_y[y_test[i]])
	y_train_origin = y_train
	y_test_origin = y_test
	y_train = np.reshape(y_train_one_hot, (len(y_train), classes, 1))
	y_test = np.reshape(y_test_one_hot, (len(y_test), classes, 1))

	return x_train, x_test, y_train, y_test, input_dim, classes, y_train_origin, y_test_origin

def Acc(output, label):
	return np.argmax(output) == np.argmax(label)

def process(file, epoch, lr):
	x_train, x_test, y_train, y_test, input_dim, classes, y_train_origin, y_test_origin = data_process(file)
	network = [
		Dense(input_dim, 5),
		Tanh(),
		Dense(5, 3),
		Tanh(),
		Dense(3, classes),
		Sigmoid()
	]
	Train_acc = train(network, mse, mse_prime, x_train, y_train, epoch, lr)
	Test_acc = test(network, mse, x_test, y_test)
	x_train = np.reshape(x_train, (len(x_train), input_dim))
	x_test = np.reshape(x_test, (len(x_test), input_dim))
	weights = network[len(network)-2].bias.ravel().tolist() + network[len(network)-2].weights.ravel().tolist()
	w_tmp = []
	for i in range(0, len(weights), 4):
		w_tmp.append(weights[i:i+4])
	weights = w_tmp
	return Train_acc, x_train, y_train_origin, Test_acc, x_test, y_test_origin, weights