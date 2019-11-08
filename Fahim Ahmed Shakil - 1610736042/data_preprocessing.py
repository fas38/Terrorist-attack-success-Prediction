import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("../global_terrorism_database_2017.csv", encoding='ISO-8859-1', low_memory=False)

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

