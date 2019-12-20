import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn import metrics
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers import BatchNormalization
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
from keras.utils.vis_utils import plot_model

# df = pd.read_csv("../global_terrorism_database_2017.csv", encoding='ISO-8859-1', low_memory=False)

df1 = pd.read_csv("gtd01.csv", encoding='ISO-8859-1')
df2 = pd.read_csv("gtd01.csv", encoding='ISO-8859-1')
df = pd.concat([df1, df2])

# feature and label
col = list(df.columns.values)
col.remove("success")
col.remove("approxdate")
col.remove("resolution")
col.remove("provstate")
col.remove("location")
col.remove("alternative_txt")
col.remove("propextent_txt")
col.remove("propcomment")
col.remove("dbsource")
col.remove("INT_LOG")
col.remove("INT_IDEO")
col.remove("INT_MISC")
col.remove("INT_ANY")
col.remove("related")
col.remove("addnotes")
df_label = df[["success"]]
df_features = df[col] #total 120 features

#categorical to number
le = LabelEncoder()

le.fit(df_features['city'].astype(str))
df_features['city'] = le.transform(df_features['city'].astype(str))

le.fit(df_features['summary'].astype(str))
df_features['summary'] = le.transform(df_features['summary'].astype(str))

le.fit(df_features['attacktype2_txt'].astype(str))
df_features['attacktype2_txt'] = le.transform(df_features['attacktype2_txt'].astype(str))

le.fit(df_features['attacktype3_txt'].astype(str))
df_features['attacktype3_txt'] = le.transform(df_features['attacktype3_txt'].astype(str))

le.fit(df_features['targsubtype1_txt'].astype(str))
df_features['targsubtype1_txt'] = le.transform(df_features['targsubtype1_txt'].astype(str))

le.fit(df_features['corp1'].astype(str))
df_features['corp1'] = le.transform(df_features['corp1'].astype(str))

le.fit(df_features['target1'].astype(str))
df_features['target1'] = le.transform(df_features['target1'].astype(str))

le.fit(df_features['natlty1_txt'].astype(str))
df_features['natlty1_txt'] = le.transform(df_features['natlty1_txt'].astype(str))

le.fit(df_features['targtype2_txt'].astype(str))
df_features['targtype2_txt'] = le.transform(df_features['targtype2_txt'].astype(str))

le.fit(df_features['targsubtype2_txt'].astype(str))
df_features['targsubtype2_txt'] = le.transform(df_features['targsubtype2_txt'].astype(str))

le.fit(df_features['corp2'].astype(str))
df_features['corp2'] = le.transform(df_features['corp2'].astype(str))

le.fit(df_features['target2'].astype(str))
df_features['target2'] = le.transform(df_features['target2'].astype(str))

le.fit(df_features['natlty2_txt'].astype(str))
df_features['natlty2_txt'] = le.transform(df_features['natlty2_txt'].astype(str))

le.fit(df_features['targtype3_txt'].astype(str))
df_features['targtype3_txt'] = le.transform(df_features['targtype3_txt'].astype(str))

le.fit(df_features['targsubtype3_txt'].astype(str))
df_features['targsubtype3_txt'] = le.transform(df_features['targsubtype3_txt'].astype(str))

le.fit(df_features['corp3'].astype(str))
df_features['corp3'] = le.transform(df_features['corp3'].astype(str))

le.fit(df_features['target3'].astype(str))
df_features['target3'] = le.transform(df_features['target3'].astype(str))

le.fit(df_features['natlty3_txt'].astype(str))
df_features['natlty3_txt'] = le.transform(df_features['natlty3_txt'].astype(str))

le.fit(df_features['gsubname'].astype(str))
df_features['gsubname'] = le.transform(df_features['gsubname'].astype(str))

le.fit(df_features['gname2'].astype(str))
df_features['gname2'] = le.transform(df_features['gname2'].astype(str))

le.fit(df_features['gsubname2'].astype(str))
df_features['gsubname2'] = le.transform(df_features['gsubname2'].astype(str))

le.fit(df_features['gname3'].astype(str))
df_features['gname3'] = le.transform(df_features['gname3'].astype(str))

le.fit(df_features['gsubname3'].astype(str))
df_features['gsubname3'] = le.transform(df_features['gsubname3'].astype(str))

le.fit(df_features['motive'].astype(str))
df_features['motive'] = le.transform(df_features['motive'].astype(str))

le.fit(df_features['claimmode_txt'].astype(str))
df_features['claimmode_txt'] = le.transform(df_features['claimmode_txt'].astype(str))

le.fit(df_features['claimmode2_txt'].astype(str))
df_features['claimmode2_txt'] = le.transform(df_features['claimmode2_txt'].astype(str))

le.fit(df_features['claimmode3_txt'].astype(str))
df_features['claimmode3_txt'] = le.transform(df_features['claimmode3_txt'].astype(str))

le.fit(df_features['weapsubtype1_txt'].astype(str))
df_features['weapsubtype1_txt'] = le.transform(df_features['weapsubtype1_txt'].astype(str))

le.fit(df_features['weapsubtype2_txt'].astype(str))
df_features['weapsubtype2_txt'] = le.transform(df_features['weapsubtype2_txt'].astype(str))

le.fit(df_features['weapsubtype3_txt'].astype(str))
df_features['weapsubtype3_txt'] = le.transform(df_features['weapsubtype3_txt'].astype(str))

le.fit(df_features['weapsubtype4_txt'].astype(str))
df_features['weapsubtype4_txt'] = le.transform(df_features['weapsubtype4_txt'].astype(str))

le.fit(df_features['weaptype2_txt'].astype(str))
df_features['weaptype2_txt'] = le.transform(df_features['weaptype2_txt'].astype(str))

le.fit(df_features['weaptype3_txt'].astype(str))
df_features['weaptype3_txt'] = le.transform(df_features['weaptype3_txt'].astype(str))

le.fit(df_features['weaptype4_txt'].astype(str))
df_features['weaptype4_txt'] = le.transform(df_features['weaptype4_txt'].astype(str))

le.fit(df_features['weapdetail'].astype(str))
df_features['weapdetail'] = le.transform(df_features['weapdetail'].astype(str))

le.fit(df_features['divert'].astype(str))
df_features['divert'] = le.transform(df_features['divert'].astype(str))

le.fit(df_features['kidhijcountry'].astype(str))
df_features['kidhijcountry'] = le.transform(df_features['kidhijcountry'].astype(str))

le.fit(df_features['ransomnote'].astype(str))
df_features['ransomnote'] = le.transform(df_features['ransomnote'].astype(str))

le.fit(df_features['hostkidoutcome_txt'].astype(str))
df_features['hostkidoutcome_txt'] = le.transform(df_features['hostkidoutcome_txt'].astype(str))

le.fit(df_features['scite1'].astype(str))
df_features['scite1'] = le.transform(df_features['scite1'].astype(str))

le.fit(df_features['scite2'].astype(str))
df_features['scite2'] = le.transform(df_features['scite2'].astype(str))

le.fit(df_features['scite3'].astype(str))
df_features['scite3'] = le.transform(df_features['scite3'].astype(str))

df_features = df_features.apply(le.fit_transform)

# # feature extraction

# 1st method
# test = SelectKBest(score_func=chi2, k=4)
# fit = test.fit(df_features, df_label)
# # summarize scores
# np.set_printoptions(precision=3)
# print(fit.scores_)
# features = fit.transform(df_features)
# # summarize selected features
# print(features)

# # 2nd method
# bestfeatures = SelectKBest(score_func=chi2, k=30)
# fit = bestfeatures.fit(df_features,df_label)
# dfscores = pd.DataFrame(fit.scores_)
# dfcolumns = pd.DataFrame(df_features.columns)
# #concat two dataframes for better visualization 
# featureScores = pd.concat([dfcolumns,dfscores],axis=1)
# featureScores.columns = ['Specs','Score']  #naming the dataframe columns
# print(featureScores.nlargest(134,'Score'))  #print 10 best features

# 3rd method - feature importance
model = ExtraTreesClassifier()
model = model.fit(df_features,df_label)
# print(model.feature_importances_)
model = SelectFromModel(model, prefit=True)
df_features = model.transform(df_features)
print(df_features.shape)
# feature_size = df_features.shape[1]


# spliting dataset
# X_train, X_test, y_train, y_test = train_test_split(df_features, df_label, 
#     test_size=0.2, stratify=df_label)
X_train, X_test, y_train, y_test = train_test_split(df_features, df_label, 
    test_size=0.2,  shuffle=True, random_state=42)

# # SMOTE
sm = SMOTE(random_state=42, ratio=1.0)
X_train, y_train = sm.fit_sample(X_train, y_train)

#PCA
# feature scaling
scaler = StandardScaler()
# Fit on training set only.
scaler.fit(X_train)
# Apply transform to both the training set and the test set.
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
# Make an instance of the Model
pca = PCA(.95)
pca.fit(X_train)
feature_size = pca.n_components_
print(pca.n_components_)
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)





# # dummy classifier
# dummy_clf = DummyClassifier(strategy = "most_frequent", random_state = 0)
# dummy_clf.fit(X_train, y_train) 
# score = dummy_clf.score(X_test, y_test) #baseline accuracy -> 0.8882467871983268 or 88.82%
# print(score)


# # define the keras model
# model = Sequential()
# # model.add(Dense(120, input_dim=feature_size, activation='relu'))
# # model.add(Dense(80, input_dim=feature_size, activation='relu'))
# # model.add(Dense(60, input_dim=feature_size, activation='relu'))
# # model.add(Dense(50, input_dim=feature_size, activation='relu'))
# # model.add(Dense(30, input_dim=feature_size, activation='relu'))
# # model.add(Dense(20, input_dim=feature_size, activation='relu'))
# # model.add(Dense(15, input_dim=feature_size, activation='relu'))
# # model.add(Dense(10, input_dim=feature_size, activation='relu'))
# # model.add(Dense(5, input_dim=feature_size, activation='relu'))
# # model.add(Dense(3, input_dim=feature_size, activation='relu'))

# # model.add(Dense(150, input_dim=feature_size, activation='relu'))
# # model.add(Dense(60, input_dim=feature_size, activation='relu'))
# # model.add(Dense(30, input_dim=feature_size, activation='relu'))

# model.add(Dense(500, input_dim=feature_size, activation='relu'))
# model.add(Dense(50, input_dim=feature_size, activation='relu'))

# model.add(Dense(1, activation='sigmoid'))

# # fully connected neural network with dropout in input layer
# model = Sequential()
# model.add(Dropout(0.2, input_shape=(feature_size,)))
# model.add(Dense(120, activation='relu', kernel_constraint=maxnorm(3)))
# model.add(Dense(30, activation='relu', kernel_constraint=maxnorm(3)))
# model.add(Dense(1, activation='sigmoid'))
# # Compile model
# sgd = SGD(lr=0.1, momentum=0.9)

# fully connected neural network with dropout in hidden layer
model = Sequential()
model.add(Dense(500, input_dim=feature_size, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(50, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(BatchNormalization())
# model.add(Dense(20, activation='relu', kernel_constraint=maxnorm(3)))
# model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
# Compile model
# sgd = SGD(lr=0.1, momentum=0.9)

# model summary
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

# fit the keras model on the dataset
history = model.fit(X_train, y_train, validation_split=0.33, shuffle=True, epochs=400, batch_size=10000)

# evaluate the keras model
_,accuracy = model.evaluate(X_test, y_test)
print('Accuracy: %.2f' % (accuracy*100))

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()