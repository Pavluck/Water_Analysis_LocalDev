# Data Gathering

## Identify data sources:
The dataset supporting the model was exported from RoboFlow, with a publically avaliable dataset has 314 colored images relevant to clean water analysis. The data is configured as a TensorFlow object format and is sized to 640 square pixels. Roboflow is a website useful for tools, datasets, and training computer vision models.
Roboflow contains open-source images. This dataset is compatiable with the project's goals: The target labels are murky and clear, which were re-purposed to be potable and non-potable for the water CNN project, other data tailoring was done to support the water analysis AI.
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
