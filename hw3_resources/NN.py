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
        self.weights = [np.random.randn(x, y) for x, y in zip(sizes[:-1], sizes[1:])]

    # def feedforward(self, a):
    #     'return output of network if input is a'
    #     temp = np.zeros(len(a))
    #     for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
    #         a = relu(np.dot(a, weight) + b)
    #
    #     # update the last one with softmax
    #     return softmax(np.dot(self.weights[-1], a) + self.biases[-1])

    def backprop(self, x, y):
        '''
            performs backpropagation with one training sample and target,
            returns the change in weights and biases for the layers
        '''
        # initialize change in weights and biases
        dweight = [np.zeros(np.shape(w)) for w in self.weights]
        dbias = [np.zeros(np.shape(b)) for b in self.biases]

        #feedforward from first layer to second last
        z_s = []
        a_s = [x]
        temp_input = x
        for l in range(self.num_layers-1):
            temp_input = np.dot(temp_input, self.weights[l]) + self.biases[l]
            z_s.append(temp_input)
            temp_input = relu(temp_input)
            a_s.append(temp_input)

        output = softmax(temp_input)

        #calculate loss
        # loss = 0
        # for i in range(len(x)):
        #     loss += -np.log(output[i][y[i]])
        # print('neural network loss---->', loss)

        #initialize changes to weights and biases
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]


        # creating y one-hot-vectors
        Y = np.array([np.zeros(self.classes) for i in range(len(y))])
        for i in range(len(y)):
            Y[i][y[i]] = 1
        print(Y)
        # output layer loss
        delta = output -  Y# change this
        nabla_w[-1] = np.mat(a_s[-2]).T * delta
        nabla_b[-1] = delta

        # compute d and delta_w for the rest of the layers and backprop
        for l in xrange(2, self.num_layers):
            print('layer --->', -l)
            delta = relu_prime(delta * np.mat(self.weights[-l+1]).T)
            nabla_w[-l] = np.mat(a_s[-l-1]).T * delta
            nabla_b[-l] = delta

        print(nabla_w)
        print(nabla_b)
        return (nabla_w, nabla_b)

    def train(self, learning_rate, epoch, X, Y):
        pass

nn = NeuralNetwork([2, 3, 3], 3)
# print(nn.weights)
# print(nn.biases)
nn.backprop(np.array([1, 1]), [2])
