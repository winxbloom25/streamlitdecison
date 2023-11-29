import pandas as pd
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
import pickle

iris = datasets.load_iris()
X = iris.data
Y = iris.target


clf = DecisionTreeClassifier()
clf.fit(X, Y)

saved_model = pickle.dumps(clf)

with open('dec_iris.pkl', 'wb') as file:
    file.write(saved_model)