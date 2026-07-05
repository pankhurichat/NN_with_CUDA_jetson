from layers import Dense
from activations import Sigmoid, Softmax

class Network:
    def __init__(self, input_size, hidden_size, output_size):
        self.dense1 = Dense(input_size, hidden_size)
        self.sigmoid = Sigmoid()
        self.dense2 = Dense(hidden_size, output_size)
        self.softmax = Softmax()

    def forward(self, x):
        x = self.dense1.forward(x)
        x = self.sigmoid.forward(x)
        x = self.dense2.forward(x)
        x = self.softmax.forward(x)
        return x
    
    def backward(self, delta):
        delta = self.softmax.backward(delta)
        delta = self.dense2.backward(delta)
        delta = self.sigmoid.backward(delta)
        delta = self.dense1.backward(delta)
        return delta
    
    def update(self, lr):
        self.dense1.update(lr)
        self.dense2.update(lr)