import random, math
import copy
import numpy as np

def relu(a):
    'returns relu of input vector'
    return np.maximum(a, 0, a)

def relu_prime(a):
    'returns relu prime'
    return np.maximum(0, a)

def relu_prime2(a):
    'returns relu prime'
    # print("########################################################")
    # print(a.shape)
    # print("########################################################")

    return np.array([1 if a[i] > 0 else 0 for i in range(a.shape[0])])

# def softmax(a):
#     # print('exp--->', a)
#     for i in range(len(a)):
#         a[i] = np.exp(a[i])/np.sum(np.exp(a[i]))
#     return a

def softmax(x):
    for i in range(len(x)):
        x[i] = x[i] - np.max(x[i])
    e = np.exp(x)
    if e.ndim == 1:
        return e / np.sum(e, axis=0)
    else: # dim = 2
        return e / np.sum(e, axis=1, keepdims=True)

# def softmax(x):
#     """Compute softmax values for each sets of scores in x."""
#     for i in range(len(x)):
#         # remove max
#         x[i] - np.max(x[i])
#     exps = np.exp(x)
#     return exps / np.sum(exps)

class NeuralNetwork(object):
    def __init__(self, sizes, classes):
        self.num_layers = len(sizes)
        self.classes = classes
        self.sizes = sizes
        # self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.biases = [np.zeros(y) for y in sizes[1:]]
        self.weights = [np.random.normal(0, 1.0/y, (x, y)) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        'return output of network if input is a'
        for weight, bias in zip(self.weights, self.biases):
            a = relu(np.dot(a, weight) + bias)

        # update the last one with softmax
        a = np.array(a)
        return softmax(a)

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
        # print("TI:", temp_input)
        for l in range(self.num_layers-1):
            temp_input = np.dot(self.weights[l].T, temp_input.T) + self.biases[l]
            z_s.append(temp_input)
            temp_input = relu(temp_input)
            # print(temp_input, 'weights layer' , l+2)
            a_s.append(temp_input)

        temp_input = np.array(temp_input)
        output = softmax(temp_input)

        #initialize changes to weights and biases
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # creating y one-hot-vectors
        Y = np.array([np.zeros(self.classes) for i in range(len(y))])
        for i in range(len(y)):
            Y[i][int(y[i])] = 1

        # output layer loss
        delta = output -  Y# change this
        # delta = np.array(delta)
        # print('activcations-- > ', a_s)
        # print('delta---->', delta)

        # nabla_w[-1] = np.mat(a_s[-2]).T * delta
        # nabla_b[-1] = np.sum(delta, axis=0)

        nabla_w[-l] = np.outer(a_s[-2], delta)
        nabla_b[-l] = delta
        # print(nabla_b[-1])

        # compute d and delta_w for the rest of the layers and backprop
        # for l in xrange(2, self.num_layers):
        #     part = delta * np.mat(self.weights[-l+1]).T
        #     delta = relu_prime(part)
        #     nabla_w[-l] = np.mat(a_s[-l-1]).T * delta
        #     nabla_b[-l] = np.sum(delta, axis=0)

        for l in xrange(2, self.num_layers):
            # part = np.dot(delta, np.mat(self.weights[-l+1]).T)
            # print("Z:", z_s)
            D = np.diag(relu_prime2(z_s[-l]))

            # print(D)
            # delta = relu_prime(part)
            W = self.weights[-l+1]
            # print("W:", W)
            # print(delta, delta.shape)
            delta = delta.reshape((delta.shape[1],))
            # print(delta, delta.shape)
            # asd
            # print(delta)
            delta = np.dot(np.dot(D, W), delta)
            # print("w4tqet")
            # print(delta)
            # asd
            # print("A:", a_s)
            nabla_w[-l] = np.outer(a_s[-l - 1], delta)
            nabla_b[-l] = delta
            # print("NW:", nabla_w[-l])
            # asd
            # nabla_w[-l] = np.dot(np.mat(a_s[-l-1]).T,delta)
            # nabla_b[-l] = np.sum(delta, axis=0)

        # print(nabla_w)
        # print(nabla_b)
        return (nabla_w, nabla_b)

    # def train(self, learning_rate, epoch, X, Y):
    # #     '''
    # #         performs SGD to train model
    # #     '''
    # #     for e in range(epoch):
    # #         for i in range(len(X)):
    # #             nabla_w, nabla_b = self.backprop(X[i], Y[i])
    # #             # update all the weights
    # #             self.weights = [x - learning_rate*y for x, y in zip(self.weights, nabla_w)]
    # #             self.biases = [x - learning_rate*y for x, y in zip(self.biases, nabla_b)]
    #     for e in range(epoch):
    #
    #         nabla_w, nabla_b = self.backprop(X, Y)
    #         self.weights = [x - learning_rate*y for x, y in zip(self.weights, nabla_w)]
    #         self.biases = [x - learning_rate*y for x, y in zip(self.biases, nabla_b)]
    #         if e%100 == 0:
    #             print('EPOCH #', e)
    #             loss = 0
    #             for i in range(len(X)):
    #                 loss += -np.log10(self.feedforward(X[i])[0][int(Y[i])])
    #             print('neural network loss---->', loss)

    def train(self, learning_rate, epoch, X, Y):
        '''
            performs SGD to train model
        '''
        for e in range(epoch):
            for i in range(len(X)):
                nabla_w, nabla_b = self.backprop(X[i], Y[i])
                # print("X:", X[i])
                # print("Y:", Y[i])
                # update all the weights
                self.weights = [x - learning_rate*y for x, y in zip(self.weights, nabla_w)]
                self.biases = [x - learning_rate*y for x, y in zip(self.biases, nabla_b)]
                # print(self.weights)
                    # calculate loss
            if e%100 == 0:
                print(self.weights)
                print('EPOCH #', e)
                loss = 0
                for i in range(len(X)):
                    loss += -np.log10(self.feedforward(X[i])[0][int(Y[i])])
                print('neural network loss---->', loss)

    def predict(self, X, Y):
        result = self.feedforward(X)
        end = np.zeros(len(X))
        for i in range(len(result)):
            end[i] = result[i].argmax()
        Y2 = Y.reshape(Y.shape[0])
        print('__________________________________')
        # print(Y2-end)
        print("ERROR:", np.sum(np.absolute(Y2-end)))
        return np.sum(np.absolute(Y2-end))


nn = NeuralNetwork([2, 10, 2], 2)
# print(nn.weights)
# print(nn.biases)
# nn.train(2, 1, [np.array([1, 1])], [2])
# nn.backprop(np.array([[1,1],[2,2]]), Y)
# print(nn.feedforward(np.array([1, 1])))

train = np.loadtxt('../../hw2_resources/data/data4_train.csv')
validate = np.loadtxt('../../hw2_resources/data/data4_validate.csv')
test = np.loadtxt('../../hw2_resources/data/data4_test.csv')

X_train = np.array(train[:,0:2])
Y_train = train[:,2:3]
Y_train = relu(Y_train)

X_validate = np.array(validate[:,0:2])
Y_validate = relu(validate[:,2:3])

X_test = np.array(test[:,0:2])
Y_test = relu(np.array(test[:,2:3]))

nn.train(0.001,500, X_train, Y_train)

print('----------------training---------------')
miss_train = nn.predict(X_train, Y_train)
print('train accuracy--> ', str((len(X_train)- miss_train)/len(X_train)))

print('----------------validate---------------')
nn.predict(X_validate, Y_validate)

print('----------------test---------------')
miss = nn.predict(X_test, Y_test)
print('test_error--->', str((len(X_test)- miss)/len(X_test)))
