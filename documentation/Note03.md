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
