import random, math
import numpy as np

def relu(a):
    'returns relu of input vector'
    return a * (a > 0)

def relu_prime(a):
    'returns relu prime'
    return np.maximum(0, a)

def softmax(a):
    exps = np.exp(a)
    return exps / np.sum(exps)

class NeuralNetwork(object):
    def __init__(self, sizes, classes):
        self.num_layers = len(sizes)
        self.classes = classes
        self.sizes = sizes
        # self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.biases = [np.zeros(y) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        'return output of network if input is a'
        temp = np.zeros(len(a))
        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            a = relu(np.dot(weight, a) + b)

        # update the last one with softmax
        return softmax(np.dot(self.weights[-1], a) + self.biases[-1])

    def backprop(self, x, y):
        # initialize change in weights and biases
        dweight = [np.zeros(np.shape(w)) for w in self.weights]
        dbias = [np.zeros(np.shape(b)) for b in self.biases]

        #feedforward from first layer to second last
        z_s = []
        a_s = []
        temp_input = x
        for l in range(self.num_layers-1):
            temp_input = np.dot(self.weights[l], temp_input) + self.biases[l]
            z_s.append(temp_input)
            temp_input = relu(temp_input)
            a_s.append(temp_input)

        output = softmax(temp_input)
        a_s.append(output)
        print('z_s', z_s)
        print('as', a_s)
        asd
        print('output', output)
        print('should be 1', np.sum(output))

        d_s = [np.zeros(np.shape(x)) for x in z_s]
        # output layer loss
        d_s[-1] = output - y
        # compute d
        for i in range(self.num_layers, -1, -1):
            pass


nn = NeuralNetwork([2, 3, 3], 3)
print(nn.weights)
print(nn.biases)
nn.backprop(np.array([1, 1]), [0,1,0])
