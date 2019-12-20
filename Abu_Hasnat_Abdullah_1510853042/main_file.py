import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression,
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn import model_selection

print("HEllo")
data = pd.read_csv("globalterrorismdb_0718dist.csv", encoding = "ISO-8859-1", low_memory = False)
x = data.iloc[:,0:133]  #independent columns
y = data.iloc[:,-1]    #target column i.e price range

bestfeatures = SelectKBest(score_func=chi2, k=25)

#apply SelectKBest class to extract top 25 best features
# bestfeatures = SelectKBest(score_func=chi2, k=25)
# fit = bestfeatures.fit(x,y)
# dfscores = pd.DataFrame(fit.scores_)
# dfcolumns = pd.DataFrame(x.columns)
# #concat two dataframes for better visualization
# featureScores = pd.concat([dfcolumns,dfscores],axis=1)
# featureScores.columns = ['Specs','Score']  #naming the dataframe columns
# print(featureScores.nlargest(25,'Score'))  #print 25 best feature

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

#Linear regression Just for FUN
predictor = LinearRegression(n_jobs=-1)
predictor.fit(X=x, y=y)
outcome = predictor.predict(X=X_TEST)
coefficients = predictor.coef_
print('Outcome : {}\nCoefficients : {}'.format(outcome, coefficients))

#SVM
X_train, X_test,Y_train, Y_test = train_test_split(x, y, test_size=0.25)
sc= StandardScaler() # StandardScaler to normally distribute the input features, both train and test data. This way the data is distributed around 0, with a standard deviation of 1.
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
classifier=SVC(kernel ='linear', C=1, gamma=1)
classifier.fit(X_train, Y_train)
y_pred= classifier.predict(X_test)
print(accuracy_score(Y_test, y_pred))

#Random Forest
rfc = RandomForestClassifier()
rfc.fit(X_train,Y_train)
rfc_predict = rfc.predict(X_test)
rfc_cv_score = cross_val_score(rfc, x, y, cv=10, scoring=’roc_auc’)
print("=== Confusion Matrix ===")
print(confusion_matrix(Y_test, rfc_predict))
print('\n')
print("=== Classification Report ===")
print(classification_report(Y_test, rfc_predict))
print('\n')
print("=== All AUC Scores ===")
print(rfc_cv_score)
print('\n')
print("=== Mean AUC Score ===")
print("Mean AUC Score - Random Forest: ", rfc_cv_score.mean())

#ANN
model = Sequential([
    Dense(12, activation='relu', input_shape=( 25 ,)),
    Dense(15, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='sgd',
              loss='binary_crossentropy',
              metrics=['accuracy'])

hist = model.fit(X_train, Y_train, batch_size=57, epochs=1000, validation_split=0.2)
#visualize the training accuracy and the validation accuracy to see if the model is overfitting
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()
pred = model.predict(X_train)
pred  = [1 if y>=0.5 else 0 for y in pred] #Threshold
print(classification_report(y_train ,pred ))
print('Confusion Matrix: \n',confusion_matrix(y_train,pred))
print()
print('Accuracy: ', accuracy_score(y_train,pred))
print()
pred = model.predict(X_test)
pred  = [1 if y>=0.5 else 0 for y in pred] #Threshold
print(classification_report(y_test ,pred ))
print('Confusion Matrix: \n',confusion_matrix(y_test,pred))
print()
print('Accuracy: ', accuracy_score(y_test,pred))
print()

#CNN
