"""
˚❀༉‧ CNN (Convolutional Neural Network)
Practice CNN Architecture
Training a CNN from a dataset of images to be able to determine potability of water from a live stream or video. 

While the model can identify visual features like clarity, 
it might struggle to accurately predict complex water parameters such as PHlevels or organic matter from an image alone. 
"""

# ~~~ Imports ~~~
import numpy as np
from scipy import signal   # signal processing functions, 2D convolution

# ~~~ CNN ~~~~
class ConvolutionalNeuralNetwork:
  """CNN, use forward and backward propergation"""

def __init__(self, input_shape, kernal, depth):
  """Since CNNs use matrices, we need to extract and anchor those features for our structure"""
  self.input_shape = input_shape  # TODO? Consider extracting by dimension
  self.kernal = kernal  # feature extraction
  self.depth = depth    # number of features we want
  # random weights for now for the convolutional layer
  self.weights = np.random.randn(self.kernal, self.kernal, self.input_shape[2], self.depth) * 0.1  # to break symmetry for normal distr.
  self.biases = np.random.randn(Self.depth) * 0.1   # random biases
  """Why Random? Upon research, this technique allows the model to effectively explore the optimization landscape to find an optimal solution (if one exists), 
  as there is no prior knowledge of the optimal weights. 
  TLDR; Prevents model from 'memorizing solutions' (since solutions are random)"""
  self.classes = 2   # potable water or non-potable water
  input_depth = self.depth * (self.input_shape[0] // 2) * (self.input_shape[1] // 2)  #  assume pooling reduces the dimensions by half
  # dense layer
  self.dense_weights = np.random.randn(input_depth, self.num_classes) * 0.1  
  self.dense_biases = np.random.randn(self.num_classes) * 0.1 
  # should be enough for now, dang.. I can see why most import existing CNN architecture
  
