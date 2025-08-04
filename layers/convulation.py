import numpy as np
from layers.layer import Layer



class Convulation(Layer):
   def __init__(self, input_shape, kernel_size, depth):
      super().__init__(self)
      input_depth, input_height, input_width = input_shape
      self.depth = depth
      self.input_shape = input_shape
      self.input_depth = input_depth
      self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
      self.kernel_shape = (depth, input_depth, kernel_size, kernel_size)
      self.kernels = np.


   def forward(self):
      return super().forward()
   
   def backward(self):
      return super().backward()