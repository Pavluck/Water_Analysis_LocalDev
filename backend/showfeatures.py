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
