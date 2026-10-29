"""
NP ✧ Updated Fall 26
CNN model for water potability classification using a ResNet18 backbone, includes data augmentation, weight decay, gradient clipping, saves misclassified images for analysis.

Version: CNNv2.7.pth.

COMPLETE
- Image augmentation
- Frozen-backbone 
- Separate backbone/head learning rates
- Weight decay and gradient clipping
- Checkpoint based on validation accuracy
TODO
- Training history and misclassified-image exports
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
from dataclasses import dataclass
from typing import Optional
import shutil
from pathlib import Path

# ~~ Load Data ~~
Training_Images = r"C:\Users\tashie\ML_Water_LocalDev\Water-Images\train"
Training_Labels = r"C:\Users\tashie\ML_Water_LocalDev\Water-Images\train\_annotations.csv"
Test_Images = r"C:\Users\tashie\ML_Water_LocalDev\Water-Images\valid"
Test_Labels = r"C:\Users\tashie\ML_Water_LocalDev\Water-Images\valid\_annotations.csv"

# ~~ Training Setup ~~
"""
Global parameters for training and dataprep 
weight decay, gradient clipping, etc
"""
Labels = {
    "potable": 0,
    "nonpotable": 1,
    "clear": 0,
    "murky": 1,
}
Classes = ("potable", "nonpotable")
Columns = ("class", "label", "potability", "clarity")
ImgNetMean = (0.485, 0.456, 0.406)
ImgNetSTD = (0.229, 0.224, 0.225)
# https://stackoverflow.com/questions/66194804/what-does-frozen-mean-for-dataclasses

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32 # increment of 16 for GPU memory
EPOCHS = 20
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4 # weight decay for regularization - prevent overfitting and improve generalization
NORMALIZATION = 1.0 # gradient clipping - prevent exploding gradients during training
NAME = "CNNv2.7.pth"
FREEZE_BACKBONE_EPOCHS = 5 # number of epochs to freeze the backbone during training
BACKBONE_LR = 0.1 # multiplied by the base learning rate to set the learning rate for the backbone parameters

@dataclass(frozen=True)
class ValidationCheckpoint:
    """Place to store the checkpoint with strongest validation result."""
    epoch: int
    validation_loss: float
    validation_accuracy: float
    path: str


class DataPrep(Dataset):
    """
    Given a directory of images and a CSV file with annotations, prepare data for training or validation. 
    Uses two columns from the CSV: one for the image filenames and one for the labels. 
    The labels used are ('potable', 'not_potable', 'clear', 'murky'), they are mapped to integer indices using a predefined mapping. 
    Added support to optional transformations for data augmentation.

    TLDR; Prepares a Dataset by mapping indices + augementation
    """
    def __init__(self, data, labels, transform=None):
        """
        Initialize the dataset with the directory of images, the CSV file with annotations, and optional transformations for data augmentation.
        """
        self.images = data
        self.transform = transform
        self.annotations = pd.read_csv(labels)
        self.targets = {'potable': 0, 'not_potable': 1, 'clear': 0, 'murky': 1}
        print(f"Loaded {len(self.annotations)} samples from {data}")
        print(f"Columns: {self.annotations.columns.tolist()}")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        """
        Obtains the image and label for a given index, applies transformations if specified, and returns the processed image and its corresponding label. 
        """
        row = self.annotations.iloc[idx]
        image = str(row.iloc[0]).strip()
        column = None
        for col in ['class', 'label', 'potability', 'clarity']:
            # Check if the column exists in the DataFrame and use it for label extraction
            if col in self.annotations.columns:
                column = col
                break
        target = str(row[column]) if column is not None else str(row.iloc[1]) 
        label = self.targets.get(target.lower(), 1) # default to non potable (1) if label not found, and convert to lowercase for case-insensitive matching
        path = os.path.join(self.images, image)
        try:
            image = Image.open(path).convert('RGB')
        except Exception as e:
            print(f"Error loading {path}: {e}")
            exit(1)

        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def get_filename(self, idx):
        return str(self.annotations.iloc[idx].iloc[0]).strip()

class WaterCNN(nn.Module):
    """
    CNN model for water potability classification using a ResNet18 backbone. The final fully connected layer is replaced with a custom head for binary classification (potable vs. non-potable). The model is initialized with pre-trained weights from ImageNet, and the final layer is modified to output the desired number of classes, 2 for binary classification.
    nn.Module is a base class for all neural network modules in PyTorch. It provides a way to define and manage layers, parameters, and forward passes in a structured manner. By subclassing nn.Module, we can create custom models with specific architectures and behaviors.
    """
    def __init__(self, num_classes=2):
        """
        Initializes the WaterCNN model with a ResNet18 backbone and a custom classification head for binary classification. The final fully connected layer is replaced with a new head consisting of a linear layer, ReLU activation, dropout for regularization, and another linear layer to output the desired number of classes. Sets up the backbone and prepares the model for training on the water potability classification task.
        """
        super().__init__() # Call the constructor of the parent class nn.Module to initialize the model and its parameters. This is necessary to properly set up the model's internal state and ensure that it can be used for training and inference.
        self.backbone = models.resnet18(pretrained=True)
        num_features = self.backbone.fc.in_features # Get the number of input features for the final fully connected layer of the ResNet18 backbone. This is used to define the new classification head that will replace the original fully connected layer
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes) # nn.Linear is used twice in the classification head: first to reduce the feature dimension from num_features to 256, and then to map the 256 features to the desired number of output classes (num_classes). The first nn.Linear layer learns a transformation of the features extracted by the backbone, while the second nn.Linear layer produces the final class scores for classification.
        )

    def forward(self, x):
        return self.backbone(x)

train_transforms = transforms.Compose([
    # Compose a series of transformations to be applied to the training images. These transformations include resizing, random horizontal flipping, random rotation, color jittering, conversion to tensor, and normalization. The purpose of these transformations is to augment the training data and improve the model's generalization ability.
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # this range was selected because it is the standard normalization range for pre-trained models in PyTorch, which are typically trained on the ImageNet dataset. The mean and standard deviation values are calculated from the ImageNet dataset and are used to normalize the input images to have similar statistical properties as the images seen during pre-training. This helps improve the model's performance and convergence during training.
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def train_epoch(model, loader, criterion, optimizer, device, clip_norm=None):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        if clip_norm is not None and clip_norm > 0:
            # the clip norm 
            nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return total_loss / len(loader), 100.0 * correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return total_loss / len(loader), 100.0 * correct / total


def main():
    print("JuneTraining: training CNNv2.5 with weight decay and grad clipping")
    training_data = DataPrep(Training_Images, Training_Labels, transform=train_transforms)
    test_data = DataPrep(Test_Images, Test_Labels, transform=val_transforms)
    train_loader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Train samples: {len(training_data)} | Val samples: {len(test_data)}")
    model = WaterCNN(num_classes=2).to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    if FREEZE_BACKBONE_EPOCHS > 0:
        for name, param in model.named_parameters():
            if name.startswith('backbone') and not name.startswith('backbone.fc'):
                param.requires_grad = False

    backbone_params = [p for n, p in model.named_parameters() if n.startswith('backbone') and not n.startswith('backbone.fc')]
    head_params = [p for n, p in model.named_parameters() if not (n.startswith('backbone') and not n.startswith('backbone.fc'))]
    optimizer = optim.Adam([
        {'params': backbone_params, 'lr': LEARNING_RATE * BACKBONE_LR, 'weight_decay': WEIGHT_DECAY},
        {'params': head_params, 'lr': LEARNING_RATE, 'weight_decay': WEIGHT_DECAY}
    ])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    history = {'train_losses': [], 'train_accuracy': [], 'test_losses': [], 'test_accuracy': []}
    checkpoint: Optional[ValidationCheckpoint] = None
    # Training loop
    for epoch in range(EPOCHS):
        if epoch == FREEZE_BACKBONE_EPOCHS:
            print(f"Unfreezing backbone at epoch {epoch+1}")
            for param in model.parameters():
                # Unfreeze all parameters in the model to allow fine-tuning of the entire network. This is done after 5 epochs to first train the new classification head while keeping the pre-trained backbone frozen. After this point, the backbone will be trained along with the head for better performance on the new task.
                param.requires_grad = True
            backbone_params = [p for n, p in model.named_parameters() if n.startswith('backbone') and not n.startswith('backbone.fc')] 
            # where n is the name of the parameter and p is the parameter itself. This line creates a list of parameters that belong to the backbone of the model, excluding the final fully connected layer (fc). The purpose of this line is to separate the parameters of the backbone from those of the classification head, so that different learning rates can be applied to them during optimization.
            head_params = [p for n, p in model.named_parameters() if not (n.startswith('backbone') and not n.startswith('backbone.fc'))]
            # Recreate the optimizer with the new parameter groups and learning rates. The backbone parameters will have a lower learning rate (BASE_LR * BACKBONE_LR_FACTOR) to prevent large updates that could disrupt the pre-trained features, while the head parameters will have the base learning rate (BASE_LR) for faster learning of the new classification task. Weight decay is applied to both parameter groups for regularization.
            
             # The head_params list will contain the parameters of the classification head, while the backbone_params list will contain the parameters of the backbone. This separation allows us to apply different optimization settings to each group of parameters.
            optimizer = optim.Adam([
                {'params': backbone_params, 'lr': LEARNING_RATE * BACKBONE_LR, 'weight_decay': WEIGHT_DECAY},
                {'params': head_params, 'lr': LEARNING_RATE, 'weight_decay': WEIGHT_DECAY}
            ])
            # Recreate the learning rate scheduler with the new optimizer. The scheduler will adjust the learning rates of both parameter groups according to the specified step size and gamma value
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE, clip_norm=NORMALIZATION)
        test_misses, test_hits = validate(model, test_loader, criterion, DEVICE)
        history['train_losses'].append(train_loss)
        history['train_accuracy'].append(train_acc)
        history['test_losses'].append(test_misses)
        history['test_accuracy'].append(test_hits)
        scheduler.step()
        # learning rates are updated for both the backbone and head parameters during training
        print(f"Epoch {epoch+1}/{EPOCHS}: train_loss={train_loss:.4f}, train_acc={train_acc:.2f}%, val_loss={test_misses:.4f}, val_acc={test_hits:.2f}%")

        # I couldn't resist
        check_checkpoint = (
            checkpoint is None
            or test_hits > checkpoint.validation_accuracy
            or (
                test_hits == checkpoint.validation_accuracy
                and test_misses < checkpoint.validation_loss
            )
        )
        if check_checkpoint:
            torch.save(model.state_dict(), NAME)
            checkpoint = ValidationCheckpoint(
                epoch=epoch + 1,
                validation_loss=test_misses,
                validation_accuracy=test_hits,
                path=NAME,
            )
            print(f"Success! Saved best validation checkpoint to {NAME}")

    if checkpoint is None:
        raise RuntimeError("Is this your first time? No validation checkpoint was produced.")

    model.load_state_dict(torch.load(checkpoint.path, map_location=DEVICE))
    best_loss, best_accuracy = validate(model, test_loader, criterion, DEVICE)
    print(
        f"\033[36m Best checkpoint: epoch={checkpoint.epoch}, "
        f"Loss={best_loss:.4f}, Accuracy={best_accuracy:.2f}%"
    )
    history['best_checkpoint'] = {
        'epoch': checkpoint.epoch,
        'validation_loss': best_loss,
        'validation_accuracy': best_accuracy,
        'path': checkpoint.path,
    }

    with open(os.path.splitext(NAME)[0] + 'history.json', 'w') as f:
        json.dump(history, f)
    print(f"Saved history to {os.path.splitext(NAME)[0] + 'history.json'}")

if __name__ == '__main__':
    print("~~~~~˚❀༉‧~~~~~~")
    main()
    print("~~~~~ finished ~~~~~~")
