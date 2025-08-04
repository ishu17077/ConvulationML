import numpy as np
from layers.layer import Layer

class Dense(Layer):
   def __init__(self, input_size, output_size):
      super().__init__()
      self.weights = np.random.randn(output_size, input_size)
      self.biases = np.random.randn(output_size, 1)


   def forward(self, input):
      super().forward()
      self.input = input
      return np.dot(self.weights, self.input) + self.biases
   
   def backward(self, output_gradient, learning_rate):
      super().backward()
      weight_gradient = np.dot(output_gradient, self.input.T)
      #? Learning rate is just a numbver or variable not a matrix so we do not need numpy
      self.weights -= learning_rate * weight_gradient 
      self.biases -= learning_rate * output_gradient
      return np.dot(self.weights.T, output_gradient)