import numpy as np
from layers.layer import Layer

class Activation(Layer):
    def __init__(self, activation, activation_prime):
        super().__init__()
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
         super().forward()
         self.input = input
         return self.activation(self.input)

    def backward(self, output_gradient, learning_rate):
        super().backward()
        return np.multiply(output_gradient, self.activation_prime(self.input))