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
    
  # TODO plot kernels    
