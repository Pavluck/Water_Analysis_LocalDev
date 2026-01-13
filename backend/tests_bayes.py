# ~~~~ NP: tests for bayes decision tree, compare accuracy with scikit learn

# -------------Necessary Imports-------------
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from bayes import DecisionTree

# ~~~~ Global Function(s) 
def accuracy(true, predicted):
  """Given the true and predicted features, compare accuracy"""
  accuracy = np.sum(true == predicted)/len(true)
  return accuracy
