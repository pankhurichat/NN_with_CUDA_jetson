import numpy as np
class Sigmoid:
    def __init__(self):
        self.output = None

    def forward(self, x):
        self.output = 1 / (1 + np.exp(-x))
        return self.output
    def backward(self, delta):
        return delta * self.output * (1-self.output)
    
class Softmax:
    def __init__(self):
        self.output=None
        
    def forward(self, x):
        exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        self.output = exp_x / np.sum(exp_x, axis=0, keepdims=True)
        return self.output

    def backward(self, delta):
        # The softmax + cross-entropy gradient is already computed as
        # (prediction - target) in CrossEntropyLoss.backward(), so here we
        # simply pass the incoming gradient through unchanged.
        return delta

