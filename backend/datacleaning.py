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
    # too many print statements... 
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

def handle_missing_values(df, strategy='mean'):
    """Handle missing values by drop or filling"""
    if strategy == 'drop':
        df = df.dropna()
    elif strategy == 'mean':
        df = df.fillna(df.mean(numeric_only=True))
    return df

def clean_data(filepath, target_column=None):
    """Given a filepath, implements data cleaning by dropping duplicates"""
    df = load(filepath)
    df = print_data(df)       # might as well see what we're doing
    df = df.drop_duplicates()
    df = handle_missing_values(df, strategy='mean')
    df = encode_categorical(df)
    
    # Separate features and target if specified
    if target_column and target_column in df.columns:
        X = df.drop(columns=[target_column])
        y = df[target_column]
        return X, y
    # pipeline it all ★
    return df
