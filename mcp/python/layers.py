import numpy as np  
class Dense:
    
    def __init__(self, input_size, output_size):
        self.x = None
        self.w = np.random.randn(output_size, input_size)*0.1
        self.b = np.zeros((output_size, 1))
        
        self.input=None
        self.z=None
        self.grad_w=None
        self.grad_b=None
    
    def forward(self, x):  
        if x.ndim == 1:
            x = x.reshape(-1,1)
        self.input = x
        if self.w.shape[1] != self.input.shape[0]:
            raise ValueError("Incompatible matrix dimensions")
        self.z = self.w @ self.input + self.b
        return self.z
    
    def backward(self, delta):
        self.grad_w = delta @ self.input.T
        self.grad_b = delta
        return self.w.T @ delta
    
    def update(self, lr):
        self.w -= lr*self.grad_w
        self.b -= lr*self.grad_b
            