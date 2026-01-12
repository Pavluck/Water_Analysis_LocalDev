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

def print_data(df):
    """Given a dataframe, displays the dataset's attributes"""
    print("Dataset shape:", df.shape)
    print("\nFirst few rows:")
    print(df.head())
    print("\nData types:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nBasic statistics:")
    print(df.describe())

def encode_categorical(df):
    """Encode categorical columns within the dataframe"""
    le = LabelEncoder()
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
    return df
