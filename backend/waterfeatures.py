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
class FeatureVis:
  """Loads the CNN, hooks to a layer to see different levels and extract visual features"""

  def __init__(self, model:nn.Module):
    """Initializes the CNN to be ready for feature exraction"""
    self.model = model
    self.features = None
    self.hook = None
    
  # hook setup (features, registers, removal) 
  def register_hook(self, layer_name:str):
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
   # TODO remove hook  
    
