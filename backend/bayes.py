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

    def __grow_tree(self, Samples, features, depth=0):
        """Grows the tree based on number of samples and features"""
        n_samples, n_features = Samples.shape
        n_labels = len(np.unique(features))
        # give stopping criterea for growing a tree
        if (depth >= self.max_depth or n_labels == 1) or n_samples < self.min_samples_split:
            # check for max depth to find a leaf node
            leaf_value = se;f._most_common_label(features)
            return Node(value=leaf_value)
        # allow the tree to grow as needed
        feat_idxs = np.random.choice(n_features, self.n_features, replace = False)
        # greedy approach for tree traversal to find most likley prediction
        best_feat, best_threst = self._best_criteria(Samples, features, feat_idxs) 
        # split tree with best feature
        left_idxs, right_idxs = self._split(Samples[:, best_feat], best_threst)
        left = self._grow_tree(Samples[left_idxs, :], features[left_idxs], depth+1)
        right = self._grow_tree(Samples[right_idxs,:], features[right_idxs], depth+1)
        
        return Node(best_feat, best_threst, left, right)

    def _best_criteria(self, Samples, features, feat_idxs): 
        """Helper function for tree traversal"""
        best_gain = -1  # go through all features
        split_index, split_thresh = None, None
        for feat_idx in feat_idxs:
            samples_column = Samples[:, feat_idx]
            thresholds = np.unique(samples_column)  # don't check the same value twice
            for threshold in thresholds:
                gain = self._information_gain(features, samples_column, threshold)
                if gain > best_gain: # update values based on one of the 7 deadly sins - greed
                    best_gain = gain    
                    split_index = feat_idx
                    split_thresh = threshold
        return split_index, split_thresh

    def _information_gain(self, features, samples_column, split_threshold):
        """Implement the Information Gain Formula """
        # entapoy of parent and weighted av of children
        parent_entropy = entropy(features)
        # gen split
        left_idxs, right_idxs = self._split(samples_column, split_threshold)
        # nothing ventured, nothing gained
        if len(left_idxs) == 0 or len(right_idxs):
            return 0 
        # calc the weighted av children's entropy
        n = len(features)
        n_left, n_right = len(left_idxs), len(right_idxs)
        ent_left, ent_right = entropy(features[left_idxs]), entropy(features[right_idxs])
        child_entropy = (n_left/n) * ent_left + (n_right/n)*ent_right
        # return the ventures gained
        igain = parent_entropy - child_entropy
        return igain

    def _split(self, samples_column, split_thresh):
        """Helper function to assist with branching"""
        left_indexes = np.argwhere(samples_column <= split_thresh).flatten() # array of conditions as a 1D vector
        right_idxs = np.argwhere(samples_column > split_thresh).flatten()
        return left_indexes, right_idxs

    def predict(self, Samples):
        """Takes the samples from the Tree and returns an array of predicted values"""
        return np.array([self._traverse_tree(sample, self.root) for sample in Samples])

    def _traverse_tree(self, sample, node):
        """Recursively traverses the tree until a leaf node is reached"""
        # base case:
        if node.reached_leaf():
            return node.value
        # recursive case: make a choice based on the prediction weights and threshold
        if sample[node.feature] <= node.threshold:
            return self._traverse_tree(sample, node.left)
        # else visit other node
        return self._traverse_tree(sample, node.right)
