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
