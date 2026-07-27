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

```
# ~~~ Imports ~~~
import pandas as pd
import os
"""
data-cleanup:
maps classes into English
bening -> potable,  keruh -> non-potable
"""

Translate = {
    "bening": "potable",
    "keruh": "non-potable"
}

Subfolders = ["train", "test", "valid"]
DataFolder = "WaterData"

for folder in Subfolders:
    csv_path = os.path.join(DataFolder, folder, "_annotations.csv")

    if os.path.exists(csv_path):
        # Read annotation CSV
        df = pd.read_csv(csv_path)

        # Replace class names using the map
        df["class"] = df["class"].map(Translate).fillna(df["class"])

        # Overwrite CSV with updated English labels
        df.to_csv(csv_path, index=False)
        print(f"Successfully mapped labels in: {csv_path}")
    else:
        # sanity check/debugging
        print(f"File not found: {csv_path}")
```

There's some error handling for the directories and subdirectores, once the code runs smoothly, the targets will be translated:
<img width="680" height="336" alt="image" src="https://github.com/user-attachments/assets/0cdabf59-437c-44ac-8271-ecd26ac2555d" />

We can verify the targets were updated by printing the head of the dataframe again:
<img width="640" height="400" alt="image" src="https://github.com/user-attachments/assets/dd572590-7ab9-4d42-a051-19618aab29e8" />

The labels can be in different formats (e.g., 'potable', 'not_potable', 'clear', 'murky'), and they will be mapped to integer indices using a predefined mapping. This also supports optional transformations for data augmentation.

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
