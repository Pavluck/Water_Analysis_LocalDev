"""
NP ˚❀༉‧ Training water quality detection model using image data.
This module handles loading water quality images from the Water-Images dataset,
training a CNN model to classify water clarity, and managing training/validation splits.
"""

# ~~~~ Necessary Imports ~~~~
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
# ~~~~ Load the Data ~~~~
"""This uses the Data from Roboflow, filepaths may need to be updated"""
Training_Images = r"ML_Water_LocalDev\Water-Images\train"
Testing = r"ML_Water_LocalDev\Water-Images\valid"
Training_labels = r"ML_Water_LocalDev\Water-Images\train\_annotations.csv"
Testing_labels = r"ML_Water_LocalDev\Water-Images\valid\_annotations.csv"

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
        imgname = str(self.data.iloc[idx]['filename'])
        imgpath = f"{self.img_dir}\\{imgname}"
        image = Image.open(imgpath).convert("RGB")

        water_class = str(self.data.ilox[idx]['class']).lower().strip()
        label = torch.tensor(self.label_encoder.transform([water_class])[0], dtype=torch.long)
        # only extract existing labels
        if self.transform:
            image = self.transform(image)
        return image, label

# TODO: ~~~~ Training & Transforming with the loaded data ~~~~~
