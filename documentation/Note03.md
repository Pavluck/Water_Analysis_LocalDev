# Data Preparation

We now have a directory of images and a CSV file with annotations, and can prepare the data for training or validation.
The CSV file(s) need at least two columns: one for the image filenames and one for the labels. More data can be added using that logic.
The labels can be in different formats (e.g., 'potable', 'not_potable', 'clear', 'murky'), such as from the last note, and they can be mapped to integer indices using a predefined mapping. 
This technique also supports optional transformations for data augmentation.

Next, we will use techniques to design, train, and deploy a NN that can analyze video inputs of a body of water (such as a river, ocean, stream, or cup) 
to extract features such as flow, color, debris, analysis etc. and return a decision for a user to see if the body of water is potable (or not).

## Load the data
The data can be loaded from a Dataset, and a class can prepare the data by mapping indices and augmentation techniques.

The class with have functions to prepare the data. The first function in our data preparation class is initialization.

The dataset, a directory with images and the CSV file with annotations, is prepared with parameters to note transformations for data augmentation.
Potable can be 0 or 1, but it must remain consistent. In this, potable water will be marked as a 1 and nonpotable will be 0. The default for water is that it is non-potable (to err on the side of caution). For this class, murky water is mapped as 0 and clear water is mapped to 1. Clear water typically is clean, while murky water from our data will be labeled as too dirty to drink.

```
# ~~~~~ imports & dependencies ~~~~~~~
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# ~~ Load Data ~~
TrainingData = "WaterData/train"
TrainingTargets = "WaterData/train/_annotations.csv"

class DataPrep(Dataset):
    """
    Takes in a Dataset, and prepares the data for training and testing by mapping indices and augmentation
    """
    def __init__(self, data, labels, transform=None):
        """
        Initialize the dataset with the directory of images, the CSV file with annotations, and parameters to note transformations for data augmentation.
        """
        self.images = data
        self.transform = transform
        self.annotations = pd.read_csv(labels)
        self.targets = {'potable': 1, 'non-potable': 0, 'clear': 1, 'murky': 0}
        # print statements for debugging
        print(f"Loaded {len(self.annotations)} samples from {data}")
        print(f"Columns: {self.annotations.columns.tolist()}")

def main():
    print("Verify the data is initialized in class function")
    train_dataset = DataPrep(TrainingData, TrainingTargets)

if __name__ == '__main__':
    main()
```

Loading the data into the class function behaves as expected:

<img width="1144" height="622" alt="image" src="https://github.com/user-attachments/assets/1c226562-780f-4526-84f4-3743e8d49977" />

## Label Extractions 
After initialization, we can create helper functions in the preparation class to return the number of samples in the dataset, obtain the image and label for a given index, apply defined transformations, and return the processed image with its corresponding label.

 The helper checks if the column exists in the DataFrame and use it for label extraction. The class defaults data to non-potable (0) if label not found, with some built in cleaning; converts to lowercase for consistency (with matching)

```
# ~~~~~ imports & dependencies ~~~~~~~
from torch.utils.data import Dataset, DataLoader

# ~~ Load Data ~~
TrainingData = "WaterData/train"
TrainingTargets = "WaterData/train/_annotations.csv"
TestData = "WaterData/test"
TestTargets = "WaterData/test/_annotations.csv"

class DataPrep(Dataset):
    """
    Takes in a Dataset, and prepares the data for training and testing by mapping indices and augmentation
    """
    def __init__(self, data, labels, transform=None):
        """
        Initialize the dataset with the directory of images (the CSV file with annotations), parameters (to note transformations for data augmentation).
        """
        self.images = data
        self.transform = transform
        self.annotations = pd.read_csv(labels)
        self.targets = {'potable': 1, 'non-potable': 0, 'clear': 1, 'murky': 0}
        print(f"Loaded {len(self.annotations)} samples from {data}")
        print(f"Columns: {self.annotations.columns.tolist()}")

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.annotations)

    def __getitem__(self, idx):
        """
        Obtains the image and label for a given index
        """
        row = self.annotations.iloc[idx]
        image = str(row.iloc[0]).strip()
        column = None
        for col in ['class', 'label', 'potability', 'clarity']:

            if col in self.annotations.columns:
                column = col
                break
        target = str(row[column]) if column is not None else str(row.iloc[1])
        label = self.targets.get(target.lower(), 0)
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
```

We can improve this by creating a safeguard when image loading fails. If there exists an image that crashes, doesn't load, or is corrupted, it would interrupt the training process.

Instead, we create a neutral gray image of the same size (for ours, it is 224x224 pixels) to maintain consistency in input dimensions.
This allows the training to continue while logging the error for later investigation.
Can't we just skip the image? Yes, but skipping images can lead to an imbalance in the dataset and may affect the training process.
By using a placeholder, we ensure that the model still receives input of the expected size and format, which can help maintain stability during training.

Alternatively, a break could be applied. Breaking the loop would stop the training process. By the placeholder, it will then log the error and continue training with the remaining images. This way, we can still utilize the majority of the dataset while being aware of any issues with specific images.

The placeholder image needs to be neutral, so it will be (224, 224), (128, 128, 128), a grey image marked as non-potable.

```
# ~~~~~ imports & dependencies ~~~~~~~
from torch.utils.data import Dataset

# ~~ Load Data ~~
TrainingData = "WaterData/train"
TrainingTargets = "WaterData/train/_annotations.csv"
TestData = "WaterData/test"
TestTargets = "WaterData/test/_annotations.csv"

class DataPrep(Dataset):
    """
    Takes in a Dataset, and prepares the data for training and testing by mapping indices and augmentation
    """
    def __init__(self, data, labels, transform=None):
        """
        Initialize the dataset with the directory of images, the CSV file with annotations, and parameters to note transformations for data augmentation.
        """
        self.images = data
        self.transform = transform
        self.annotations = pd.read_csv(labels)
        self.targets = {'potable': 1, 'non-potable': 0, 'clear': 1, 'murky': 0}
        print(f"Loaded {len(self.annotations)} samples from {data}")
        print(f"Columns: {self.annotations.columns.tolist()}")


    def __getitem__(self, idx):
        """
        Obtains the image and label for a given index, applies transformations if specified, and returns the processed image with  label.
        If the image cannot be loaded, a placeholder image is returned instead.
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
        label = self.targets.get(target.lower(), 0)
        # default to non potable (0) if label not found, and convert to lowercase for case-insensitive matching
        path = os.path.join(self.images, image)
        try:
            image = Image.open(path).convert('RGB')
        except Exception as e:
            print(f"Error loading {path}: {e}")
            image = Image.new('RGB', (224, 224), (128, 128, 128))
        if self.transform is not None:
            image = self.transform(image)
        return image, label

```
