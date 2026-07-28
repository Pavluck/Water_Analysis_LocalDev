# Data Preparation

We now have a directory of images and a CSV file with annotations, and can prepare the data for training or validation.
The CSV file(s) need at least two columns: one for the image filenames and one for the labels. More data can be added using that logic.
The labels can be in different formats (e.g., 'potable', 'not_potable', 'clear', 'murky'), such as from the last note, and they can be mapped to integer indices using a predefined mapping. 
This technique also supports optional transformations for data augmentation.

Next, we will use techniques to design, train, and deploy a NN that can analyze video inputs of a body of water (such as a river, ocean, stream, or cup) 
to extract features such as flow, color, debris, analysis etc. and return a decision for a user to see if the body of water is potable (or not).

## Load the data
The data can be loaded from a Dataset, and a class can prepare the data by mapping indices and augmentation techniques.

```
# ~~~~~ imports & dependencies ~~~~~~~
from torch.utils.data import Dataset, DataLoader

# ~~ Load Data ~~
TrainingData = "WaterData/train"
TrainingTargets = "WaterData/train/_annotations.csv"
TestData = "WaterData/test"
TestTargets = "WaterData/test/_annotations.csv"
```
