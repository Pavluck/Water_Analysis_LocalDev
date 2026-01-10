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
#~~~~ Node Class
class Node:
    """Helper class to assist in loading the data into a tree"""
    def __init__(self, feature=None, threshold=None, left=None, right=None,*,value=None):
        """Initialize the parameters, the * passes non-key args"""
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def fit(self, Samples, features):
        """make sure the tree does not grow larger than the number of existing features"""
        self.n_features = Samples.shape[1] if not self.n_features else min(self.n_features, Samples.shape[1])
        # allows tree to grow
        self.root = self._grow_tree(Samples, features)
        return
