"""
☆ NPavelek ☆
Show CNN features extracted from video frames. 
takes the processed video data, runs it through a CNN model, and visualizes the extracted features for analysis.
Display feature map and kernels 
"""
# ~~~~ Necessary Imports ~~~~
import sys
import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pathlib import Path
from testvideostream import load_water_model, test_transforms

# ~~~~ Global Constant ~~~~~
Directory = Path(__file__).resolve().parent
"""
ensures the scripts can find own directory 
regardless of from where the command is executed.
"""
if str(Directory) not in sys.path:
  # safegaurd to prevent duplicate entries in path
  sys.path.insert(0, str(Directory)

# ~~~~ Class Functions ~~~~~                 
class Visuals:
  """Loads the CNN, hooks to a layer to see different levels and extract visual features"""

  def __init__(self, model:nn.Module):
    """Initializes the CNN to be ready for feature exraction"""
    self.model = model
    self.features = None
    self.hook = None
    
  # hook setup (features, registers, removal) 
  def hook_register(self, layer_name:str):
    """
    Hook a layer to see different levels of feature extraction
    Reference: https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook
    """
    for name, module in self.model.named_modules():
      if name == layer_name:
        self.hook = module.register_forward_hook(self._hook_fn)
        print(f"Hooked to layer: {layer_name}")
        return
    raise ValueError(f"Error: {layer_name} not found")
  
  def hook_remove(self):
    """Frees the model of the hook"""
    if self.hook is not None:
      self.hook.remove()
      self.hook = None
    
  # ~~~~ ✧ CNN Visuals ✧ ~~~~
  def plot_features(self, max_features: int=64, cmap:str = 'viridis'):
    """ 
    Viridis highlights different patterns and activations in the CNN features map
    To quote: "Pretty, oh so pretty"
    Reference: https://matplotlib.org/stable/tutorials/colors/colormaps.html
    """
    if self.features is None:
      return
    # ~~~ Setup ~~~
    num_features = self.features.shape[1]
    num_visualize = min(num_features, max_features)
    grid_size = int(np.ceil(np.sqrt(num_visualize)))
    plt.feature(figsize=(grid_size*2, grid_size*2))
    # ~~~ Plot loop (plot features one by one) ~~~
    for i in range(num_visualize):
      plt.subplot(grid_size, gride_size, i+1)
      # use imshow to display data as a 2D image
      plt.imshow(self.features[0,i],cmap=cmap)
      plt.axis('off')
    plt.subtitle("Water Analysis ~✧~ Feature Map")
    plt.tight_layout()
    plt.show()
    
    
  def plot_kernels(model: nn.Module, max_kernels: int=16):
    """
    Each kernel is visualized as an image, where the intensity of each pixel corresponds to the weight value of that kernel. 
    Reference: https://pytorch.org/docs/stable/nn.html#torch.nn.Conv2d
    """
    # ~~~ sanity check ~~~
    if not hasattr(model, 'backbone'):
      return
    conv_layer = getattr(model.backbone, 'conv1', None)
    if conv_layer is None:
      return
    weights = conv_layer.weight.detach().cpu().numpy()
    num_kernels = min)weights.shape[0], max_kernels)
    grid_size = int(np.ceil(np.sqrt(num_kernels)))
    plt.figure(figsize=(grid_size * 2, grid_size * 2))
    for i in range(num_kernels):
      kernel = weights[i]
      # extract color channel from the final dimension
      kernel = np.transpose(kernel, (1, 2, 0))
      # normalize kernel value
      kernel = (kernel - kernel.min()) / kernel.max() - kernel.min() + 1e-9)
      plt.subplot(grid_size, grid_size, i + 1)
      plt.imshow(kernel)
      plt.axis('off')
      plt.tight_layout()
      plt.show

  
  def preprocess_frame(frame, transform):
    """
    Given a BGR frame returns a normalized tensor batch
    """
    # CNNS expect 4D tensor (Size, Channels, Height, Width)
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # 0th dimension to batch/stack the image
    return transform(image).unsqueeze(0)

  # TODO Visual CNN features from video/file
  def run_stream(model: nn.Module, layer_name: str = 'backbone.layer1', source=0, max_frames: int =1):
    """
    Hook to file
    """
    feature_visualizer = FeatureVisualizer(model)
    feature_visualizer.register_hook(layer_name)
    cap = sv2.VideoCapture(source)
    # sanity check
    if not cap.isOpened():
      raise RuntimeError(f"Oopsie Daisy~ Unable to open video source: {source}")
    # TODO while loop and functionality
