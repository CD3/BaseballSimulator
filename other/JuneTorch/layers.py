import numpy as np



# functions

def sigmoid(x):
    return 1 / (1 + np.exp(-x)) 


def squared_mean_loss(y, t):
    # y and t: row=instance, column=nodes
    # return will be a 1-dim array with len(return)_= len(y)
    return -0.5 * np.sum((y-t)**2, axis=1) 


def cross_entropy_error(y, t):
    batch_size = len(y)
    cost = (np.log(y) * t) + (np.log(1-y) * (1-t))
    cost *= -1/batch_size
    cost = np.sum(cost)
    return cost 


def softmax(x):
    # x: row=instance :: column=classes=nodes
    x_exp = np.exp(x)
    sum_exp = np.sum(x_exp, axis=1)
    sum_exp = np.array([sum_exp]) # row=1 :: column=sum for each instance
    probabilities = x_exp / sum_exp.T
    return probabilities


def softmax_loss(x, t):
    y = softmax(x)
    return cross_entropy_error(y, t)



# Layers

class Affine:
    def __init__(self, params, bias):
        self.params = params # row=next layer :: column=current layer
        self.bias = bias # row=1 :: column=next layer
        self.input = None # row=instance :: column=feature
        self.weigted_sum = None

        self.d_params = None
        self.d_bias = None
        self.d_input = None

    def forward(self, input_from_prev_layer):
        self.input = input_from_prev_layer # row=instance :: column=feature
        self.weigted_sum = np.dot(self.params, self.input.T) + self.bias.T # row=nodes :: column=instance
        self.weigted_sum = self.weigted_sum.T # row=instance :: column=nodes
        return self.weigted_sum # row=instance :: column=nodes

    def backward(self, d_output):
        # d_output --> row=instance :: column=next layer
        self.d_input = np.dot(d_output, self.params) # row=instance :: column=current layer
        self.d_params = np.dot(d_output.T, self.input) # row=next layer :: column=current layer
        self.d_bias = np.sum(d_output, axis=0) # row=1 :: column=next layer

        return self.d_input


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx


class LeakyRelu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0 )
        out = x.copy()
        out[self.mask] = 0.01 * out[self.mask]

        return out
    
    def backward(self, dout):
        dout[self.mask] = 0.01 * dout
        dx = dout

        return dx 


class LeakyReluWithLoss:
    def __init__(self):
        self.mask = None
        self.loss = None
        self.batchsize = None
        self.out = 0

    def forward(self, x, labels):
        self.batchsize = len(x)
        self.mask = (x <= 0 )
        self.out = x.copy()
        self.out[self.mask] = 0.01 * self.out[self.mask]
        self.loss = squared_mean_loss(self.out, labels) / self.batchsize
        return self.out
    
    def backward(self, dout):
        dout[self.mask] = 0.01 * dout
        dx = dout

        return dx 


class SoftmaxWithLoss:
    def __init__(self):
        self.input = None # row=instance :: column=nodes
        self.processed = None # row=instance :: column=nodes
        self.loss = None # row=0 :: column=nodes
        self.labels = None # row=1 column=classes=nodes
        self.batchsize = None

    def forward(self, predictions, labels):
        self.batchsize = len(predictions)
        self.input = predictions # row=instance :: column=nodes
        self.labels = labels # row=1 column=classes=nodes
        self.processed = softmax(predictions)
        self.loss = cross_entropy_error(self.processed, labels)

    def backward(self, d_output):
        dx = self.processed - self.labels
        dx /= self.batchsize
        return dx

