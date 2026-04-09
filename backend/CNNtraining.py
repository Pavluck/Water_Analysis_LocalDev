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
transition computations from the CPU to the GPU 
checks if a compatible NVIDIA GPU is available and sets the device accordingly. If a GPU is present, it will use it for training the model

Reference: https://github.com/opencv/opencv/issues/20227

TLDR; allows the code to run efficiently on systems with a GPU while still being compatible with those without one.
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batchSize = 32
Epoches = 20
LEARNING_RATE = 0.001
CNNFilePath = "water_potability.pth"

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
        # Load Labels to identify potatbility Targets
        self.annotations = pd.read_csv(target)
        # Train that Clear water is more likely to be potable
        self.class_to_idx = {
            'potable':0,
            'not_potable':0,
            'clear':0,
            'murky':1
        }
        # helper prints (for training details/debugging) to see what CNN is getting
        print(f"Loaded {len(self.annotations)} samples from {images}")
        print(f"Number of Columns/Features: {self.annotations.columns.tolist()}")
        print(f"Number of Rows/Images: {self.annotations.rows.tolist()}")

    def __len__(self):
        """Minor helper function. Returns the length/dimension of the target"""
        return len(self.annotations)
    
    def __getitem__(self, idx):
        """Used by DataLoader to determine the size of the dataset with the given index.
        Applies transformations/preprocessing to return the image/label as tensors
        """
        # Get image filename (typically in first column after Roboflow export)
        row = self.annotations.iloc[idx]
        image_filename = row.iloc[0]  # First column is usually the filename
        
        label_col = None
        for col in ['class', 'label', 'potability', 'clarity']:
            if col in self.annotations.columns:
                label_col = col
                break
        
        if label_col is None:
            # If no standard column found, use the second column
            label_str = row.iloc[1]
        else:
            label_str = row[label_col]
        
        # Convert string label to index
        label = self.class_to_idx.get(str(label_str).lower(), 0)
        
        # Load and process image
        image_path = os.path.join(self.images_dir, str(image_filename).strip())
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Oopsie Daisy~ Error loading image {image_path}: {e}")
            # Return a blank image if loading fails
            image = Image.new('RGB', (224, 224))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# ~~~~ CNN Model for Image Classification ~~~~
class WaterCNN(nn.Module):
    """reads data (image_tensor, label), builds CNN for classification for water potability"""

    def __init__(self, num_classes=2):
        """Anchor the labels, images, the transformation and label type"""
        super().__init__()

        # Use ResNet18's pretrained backbone
        # a standard approach for transfer learning, 
        # where the model is initialized with weights trained on ImageNet
        self.backbone = models.resnet18(pretrained=True)
        # now modify the fully connected layer from the backbone; prepares for binary classification
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, data):
        """Forward propogation, returns the feature map given the input tensor data"""
        return self.backbone(data)

    def __getitem__(self, index):
        """ Using an index from the dataframe, PyTorch matches it to fetch the image data and labels for a specific index during training. """
        image = self.images[index]
        label = self.label[index]
        # only extract existing labels
        if self.transform:
            image = self.transform(image)
        return image, label

# TODO: Finish Training & Transforming with the loaded data ~~~~~
# ~~~~ Image Transforms ~~~~
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# The transform for validation are defined separately
# moving average of the mean and variance learned during the training phase to normalize activations
validation_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
