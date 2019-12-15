import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

print("HEllo")
data = pd.read_csv("globalterrorismdb_0718dist.csv", encoding = "ISO-8859-1", low_memory = False)
x = data.iloc[:,0:133]  #independent columns
y = data.iloc[:,-1]    #target column i.e price range

bestfeatures = SelectKBest(score_func=chi2, k=25)

#apply SelectKBest class to extract top 25 best features
bestfeatures = SelectKBest(score_func=chi2, k=25)
fit = bestfeatures.fit(x,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(x.columns)
#concat two dataframes for better visualization
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(25,'Score'))  #print 25 best feature

# Feature Selection
model = ExtraTreesClassifier()
model.fit(x,y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=x.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()

#K-Fold Cross Validation
model = svm.SVC()
accuracy = cross_val_score(model, x, y, scoring='accuracy', cv = 10)
print(accuracy)
#get the mean of each fold
print("Accuracy of Model with Cross Validation is:",accuracy.mean() * 100)

#Linear regression
predictor = LinearRegression(n_jobs=-1)
predictor.fit(X=x, y=y)
outcome = predictor.predict(X=X_TEST)
coefficients = predictor.coef_
print('Outcome : {}\nCoefficients : {}'.format(outcome, coefficients))