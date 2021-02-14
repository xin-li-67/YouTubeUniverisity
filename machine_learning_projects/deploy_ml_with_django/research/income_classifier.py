import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split    # will be used for data split
from sklearn.preprocessing import LabelEncoder          # for preprocessing
from sklearn.ensemble import RandomForestClassifier     # for training the algorithm
from sklearn.ensemble import ExtraTreesClassifier       # for training the algorithm
import joblib                                           # for saving algorithm and preprocessing objects

# LOAD DATASET
df = pd.read_csv('https://raw.githubusercontent.com/pplonski/datasets-for-start/master/adult/data.csv', skipinitialspace=True)
x_cols = [c for c in df.columns if c != 'income']

X = df[x_cols]
Y = df['income']
# df.head()

# SPLIT DATA AND FILL IN MISSING VALUE
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1234)
train_mode = dict(X_train.mode().iloc[0])
X_train = X_train.fillna(train_mode)
print(train_mode)

# PREPROCESSING: convert categoricals
encoders = {}
for column in ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex','native-country']:
    categorical_convert = LabelEncoder()
    X_train[column] = categorical_convert.fit_transform(X_train[column])
    encoders[column] = categorical_convert

# Random Forest algorithm
rf = RandomForestClassifier(n_estimators = 100)
rf = rf.fit(X_train, Y_train)

# Extra Trees algorithm
et = ExtraTreesClassifier(n_estimators = 100)
et = et.fit(X_train, Y_train)

# save preprocessing objects and algorithm
joblib.dump(train_mode, "./train_mode.joblib", compress=True)
joblib.dump(encoders, "./encoders.joblib", compress=True)
joblib.dump(rf, "./random_forest.joblib", compress=True)
joblib.dump(et, "./extra_trees.joblib", compress=True)