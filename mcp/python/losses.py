import numpy as np
class CrossEntropyLoss:
    def __init__(self):
        self.prediction = None
        self.target = None
        
    def forward(self, prediction, target):
        prediction = np.clip(prediction, 1e-15, 1 - 1e-15)
        self.prediction = prediction
        self.target = target
        loss = -np.sum(target * np.log(prediction))
        return loss
    
    def backward(self):

        return self.prediction - self.target