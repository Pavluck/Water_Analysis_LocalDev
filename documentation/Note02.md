# Data Gathering

## Source:
The dataset supporting the model was exported from RoboFlow, with a publicly available dataset has 314 colored images relevant to clean water analysis. The data is configured as a TensorFlow object format and is sized to 640 square pixels. Roboflow is a website useful for tools, datasets, and training computer vision models.
Roboflow contains open-source images. This dataset is compatible with the project's goals: The target labels are murky and clear, which were re-purposed to be potable and non-potable for the water CNN project, other data tailoring was done to support the water analysis AI.
Additionally, the dataset used from RoboFlow has a CC 4.0 license and includes machine learning splits.

Other miscellaneous data images were hand picked for the project, with labels constructed manually. The images found were open sourced and publically obtained. However, the hand picked images are a minor subset of the data used, most of the model was built using the Roboflow dataset. 

## Obtaining Data:
An API key was used with the Roboflow python library to obtain the workspace data. Running the code requires an extra dependency as well as configuring an API key from their website.
```
"""
Python code ~ Obtaining data from Roboflow 
"""
from roboflow import Roboflow        # library/dependency 
rf = Roboflow(api_key="✧~‧˚❀༉‧˚")    # <-- api placeholder text for security
project = rf.workspace("waterqualityprediction").project("water-quality-prediction")
version = project.version(2)
dataset = version.download("tensorflow")
```
Once you have the API key configured and run the code, you will see a similar output:
<img width="736" height="286" alt="image" src="https://github.com/user-attachments/assets/43fa23d7-df99-4261-a37c-d8e604b67df7" />

Since the Roboflow Python SDK, download() method accepts a location parameter, it can be used to specify the destination directory (an easier to type folder name) for the downloaded dataset. 

```
dataset = version.download("tensorflow", location="WaterData")
```

In the future, the WaterData directory will be referenced as the directory that holds the Roboflow data.

## Data Cleanup
After the data is gathered, we now have a directory of images and a CSV file with annotations. This can be used to prepare the data for training and eventually, validation.
The CSV file should have at least two columns: one for the image filenames and one for the labels.

```
import pandas as pd
filepath = "WaterData/test/_annotations.csv"
df = pd.read_csv(filepath)
print(df[["filename", "class"]].head(10))
```

The code will display the data like so:
<img width="550" height="244" alt="image" src="https://github.com/user-attachments/assets/fbe964db-cb4b-4685-a0b4-0e695dfb24f5" />

The class names, aka, our labels are annotated in Indonesian. Keruh is a word for murky and bening is for clear. We will clean up the data from the test, train, and validate folders to be mapped to potable and non-potable (respectively).

## References:
@misc{ water-quality-prediction_dataset,
  title = { Water-Quality-Prediction Dataset },
  type = { Open Source Dataset },
  author = { WaterQualityPrediction },
  howpublished = { \url{ https://universe.roboflow.com/waterqualityprediction/water-quality-prediction } },
  url = { https://universe.roboflow.com/waterqualityprediction/water-quality-prediction },
  journal = { Roboflow Universe },
  publisher = { Roboflow },
  year = { 2024 },
  month = { jul },
  note = { visited on 2026-07-23 },
}
