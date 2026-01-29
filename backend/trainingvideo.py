"""
NP ˚❀༉‧ Training water quality detection model using image data.
This module handles loading water quality images from the Water-Images dataset,
training a CNN model to classify water clarity, and managing training/validation splits.
"""

# ~~~~ Necessary Imports ~~~~
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
