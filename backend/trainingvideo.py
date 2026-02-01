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
# ~~~~ Load the Data ~~~~
Training_Images = r"ML_Water_LocalDev\Water-Images\train"
Testing = r"ML_Water_LocalDev\Water-Images\valid"
Training_labels = r"ML_Water_LocalDev\Water-Images\train\_annotations.csv"
Testing_labels = r"ML_Water_LocalDev\Water-Images\valid\_annotations.csv"

class WaterQualityDataset(Dataset):
    """reads data from CSV and returns (image_tensor, label)."""
    Potable_labels = {'clear': 'potable', 'murky': 'not_potable'}

    def __init__(self, csv_file, img_dir, transform=None, label_mode='clarity'):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.label_mode = label_mode

        if label_mode == 'potability':
            self.data['class'] = self.data['class'].str.lower().str.strip().map(self.Potable_labels)

        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.data['class'].unique())
      
