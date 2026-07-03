"""
NP ˚✷‧ Water AI: June CNN Model Training
New training revision with weight decay and gradient clipping

Loads the training data (from Roboflow), training, then saves a CNN model to classify water potability.
The model is a ResNet18 backbone (custom fully connected layer)
Includes data augmentation for training and a learning rate scheduler for better convergence.
+ evaluation on the validation set and saves misclassified images for analysis.


References: 
https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
"""

# ~~~~~ imports & dependencies ~~~~~~~
import os
import json
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms, models

# ~~ Load Data ~~
Training_Images = r"\ML_Water_LocalDev\Water-Images\train"
Training_Labels = r"\ML_Water_LocalDev\Water-Images\train\_annotations.csv"
Test_Images = r"\ML_Water_LocalDev\Water-Images\valid"
Test_Labels = r"\ML_Water_LocalDev\Water-Images\valid\_annotations.csv"

# ~~ Global Constants ~~
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32  # increment of 16 for GPU optimization
EPOCHS = 20
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
NORMALIZATION = 1.0  # gradient clipping limits max value for grads during backward propergation
NAME = "CNNv2.5.pth" 
BACKBONE_FREEZE = 5 # unfreeze backbone every 5 epoches
BACKBONE_LR = 0.1  # multiplied by base LR 

class DataPrep(Dataset):
  """
  Takes in a Dataset, and prepares the data by mapping indices and augementation
  """
  def __init__(self, data, labels, transform=None):
    """
    Initializes the dataset of images, targets, and transformation (augmentation)
    """
    self.images = data
    self.transform = transform
    self.annotations = pd.read_csv(labels)
    self.targets = {'potable':0, 'not_potable':1, 'clear':0, 'murky':1}

  def __indexmap(self, index):
    """
    Obtains image with its cooresponding label
    """
    row = self.annotations.iloc[index]
    image = str(row.iloc[0]).strip()
    columnn = None
    
    # Check if the column exists in the DataFrame before using for label extraction
    for col in ['class', 'label', 'potability', 'clarity']:
      if col in self.annotations.columns:
        column = col
        break
  
