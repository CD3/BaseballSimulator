import numpy as np
from layers import *    # ignore unused wildcaart import warning
from collections import OrderedDict



class BasicNet:
    def __init__(self, in_size=6, hidden_size=50, out_size=3):
        fitter = 0.01

        self.params = {}
        self.params['W1'] = np.random.randn(hidden_size, in_size) * fitter
        self.params['B1'] = np.random.randn(1, hidden_size)
        self.params['W2'] = np.random.randn(out_size, hidden_size) * fitter
        self.params['B2'] = np.random.randn(1, out_size) 

        self.layers = OrderedDict()
        self.layers['1_Affine'] = Affine(self.params['W1'], self.params['B1'])
        self.layers['1_Relu'] = Relu()
        self.layers['2_Affine'] = Affine(self.params['W2'], self.params['B2'])
        self.layers['2_LeakyRelu'] = LeakyReluWithLoss()

    # if cost isn't needed as a return, set train_y=0  
    def forward(self, train_x, train_y):
        x = train_x
        for layer in self.layers.values():
            try:
                x = layer.forward(x)
            except:
                layer.forward(x, train_y)

        probs = x
        loss = self.layers['2_LeakyRelu'].loss
        return {'probs': probs, 'loss':loss}

    def backward(self, train_x, train_y):
        self.forward(train_x, train_y)

        layers = list(self.layers.values())
        layers.reverse()

        d_output = 1
        for layer in layers:
            d_output = layer.backward(d_output)

        grads = {}
        grads['W1'] = self.layers['1_Affine'].d_params
        grads['B1'] = self.layers['1_Affine'].d_bias
        grads['W2'] = self.layers['2_Affine'].d_params
        grads['B2'] = self.layers['2_Affine'].d_bias

        return grads

    def accuracy(self, test_x, test_y):
        pass
