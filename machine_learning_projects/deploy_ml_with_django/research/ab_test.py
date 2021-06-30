import json
import requests
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

# LOAD DATASET
df = pd.read_csv('https://raw.githubusercontent.com/pplonski/datasets-for-start/master/adult/data.csv', skipinitialspace=True)
x_cols = [c for c in df.columns if c != 'income']
X = df[x_cols]
Y = df['income']
# df.head()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state=1234)

# use the first 100 rows of the test data for A/B Test
for i in range(100):
    input_data = dict(X_test.iloc[i])
    target = Y_test.iloc[i]
    r = requests.post("http://127.0.0.1:8000/api/v1/income_classifier/predict?status=ab_testing", input_data)
    response = r.json()
    # provide feedback
    requests.put("http://127.0.0.1:8000/api/v1/mlrequests/{}".format(response["request_id"]), {"feedback": target})