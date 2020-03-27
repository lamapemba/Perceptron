import numpy as np 

training_sets_inputs = np.array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])
training_sets_outputs = np.array([[0,1,1,0]]).T


def sigmoid(x):
	return 1/(1 + np.exp(-x))


def sigmoid_derivatives(x):  #for back propagation
	return x * (1 - x)

np.random.seed(1)
synaptic_weights = 2 * np.random.random((3, 1)) - 1

print('Random starting synaptic weights: ')
print(synaptic_weights)

for i in range(10000):
	input_layer = training_sets_inputs
	outputs = sigmoid(np.dot(input_layer, synaptic_weights))
	error = training_sets_outputs - outputs
	adjustments = error * sigmoid_derivatives(outputs)

	synaptic_weights += np.dot(input_layer.T, adjustments)

print('Synaptic weights after training: ')
print(synaptic_weights)
print('Outputs after training: ')
print(outputs)