"""
NP ˚❀༉‧ Water AI CNN model 
Loads the training data (from Roboflow), training, then saves a CNN model to classify water potability.
"""

# ~~~~ Necessary Imports ~~~~
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms, models
import matplotlib.pyplot as plt

# ~~~~ Load the Data ~~~~
"""This uses the Data from Roboflow, filepaths may need to be updated"""
Training_Images = r"ML_Water_LocalDev\Water-Images\train"
Testing = r"ML_Water_LocalDev\Water-Images\valid"
Training_labels = r"ML_Water_LocalDev\Water-Images\train\_annotations.csv"
Testing_labels = r"ML_Water_LocalDev\Water-Images\valid\_annotations.csv"

# ~~~~ Hyperparameters and Configurations ~~~~
"""
creates a symbolic handle representing an NVIDIA GPU. It is the standard way to transition computations from the CPU to the GPU 
checks if a compatible NVIDIA GPU is available and sets the device accordingly. If a GPU is present, it will use it for training the model

Reference: https://github.com/opencv/opencv/issues/20227

TLDR; allows the code to run efficiently on systems with a GPU while still being compatible with those without one.
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batchSize = 32
Epoches = 20
LEARNING_RATE = 0.001
CNNFilePath = "water_potability_image_model.pth"

# ~~~~ Class for Loading Roboflow Dataset into Pytorch ~~~~
class RoboflowData(Dataset):
    """Pytorch works well with Roboflow data
    https://docs.pytorch.org/docs/stable/cuda.html"""
    def __init__(self, images, target, trainsform=None):
        """ 
        Ues the directory of images and the CSV file containing labels to prepare the dataset for training.
            images_dir: Directory containing images
            labels_csv: Path to CSV with image filenames and labels
            transform: Image transformations to apply (None for now)
        """
        self.images = images
        self.transform = transform
        
class WaterQualityDataset:
    """reads data from CSV and returns (image_tensor, label)."""
    Potable_labels = {'clear': 'potable', 'murky': 'not_potable'}

    def __init__(self, labels, images, transform=None, label_mode=None):
        """Anchor the labels, images, the transformation and label type"""
        self.labels = labels
        self.images = imgages
        self.transform = transform
        self.label_mode = label_mode

        if label_mode == 'potability':
            self.data['class'] = self.data['class'].str.lower().str.strip().map(self.Potable_labels)

        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.data['class'].unique())

    def __len__(self):
        """Returns the total number of samples in the dataset"""
        return len(self.images)

    def __getitem__(self, index):
        """ Using an index from the dataframe, PyTorch matches it to fetch the image data and labels for a specific index during training. """
        image = self.images[index]
        label = self.label[index]
        # only extract existing labels
        if self.transform:
            image = self.transform(image)
        return image, label

# TODO: ~~~~ Finish Training & Transforming with the loaded data ~~~~~
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.2,0.2),
    transforms.ToTensor(),
    # Imagenet pretrained model uses this mean and stD:
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    # TODO: Recalculate for water images
])

# The transform for validation are defined separately
# moving average of the mean and variance learned during the training phase to normalize activations
validation_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
