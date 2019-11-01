import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.preprocessing import LabelEncoder

gtd = pd.read_csv("../global_terrorism_database_2017.csv", encoding='ISO-8859-1', low_memory=False)
print(gtd.head(20))

