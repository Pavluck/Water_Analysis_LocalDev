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

# ------------- Load & Test -------------
data = datasets.load_breast_cancer()
Samples = data.data
features = data.target

Samples_train, Samples_test, feat_train, feat_test = train_test_split(Samples, features, test_size=0.2, random_state=1234)

clf = DecisionTree(max_depth=10)
# create decision with built tree
clf.fit(Samples_train, feat_train)

feat_prediction = clf.predict(Samples_test)
acc = accuracy(feat_test, feat_prediction)

print("Accuracy: ", acc)
# turns out to be about ~ 6052631578947368
# scikit learn's is better, closer to 0.95
