# ~~~~ NP - Data Cleaning ~~~~
"""
˚❀༉‧ Clean and preprocess water analysis data
"""
# ~~~~ Imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

# ~~~~ Global Functions
def load(filepath):
  """Given a filepath to a CSV file, returns the dataframe"""
  df = pd.read_csv(filepath)
  return df
  
