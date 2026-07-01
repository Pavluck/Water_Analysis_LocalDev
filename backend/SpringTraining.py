"""
NP ˚✷‧ Water AI: June CNN Model Training
New training revision with weight decay and gradient clipping to produce CNNv2.5.pth.

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
