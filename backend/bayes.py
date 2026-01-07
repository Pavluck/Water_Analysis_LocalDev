""" NP ‧˚❀༉‧˚. Bayes Decision Tree
Loads & Fit the Data into a Decision tree
Builds to the input data so that the tree can be populated
Classification Method (the goal is to predict discrete labels for potability)
"""
#~~~~ Necessary Imports
import numpy as np
from collections import Counter
#~~~~ Global Functions
def entropy(features):
    """This method calculates the measure of uncertainty"""
    histogram = np.bincount(features) # calculates number of occurrences of class labels
    prob = histogram / len(features)  # histogram 
    uncertainty = -np.sum([p * np.log2(p) for p in prob if p > 0])
    return uncertainty
