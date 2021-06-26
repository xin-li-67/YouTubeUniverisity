# supervised learning algorithm
import sklearn
import pandas as pd

from sklearn import preprocessing, model_selection
from sklearn.neighbors import KNeighborsClassifier

# prepare the data
data = pd.read_csv("./datasets/car_dataset/car.data")
# convert data
le = preprocessing.LabelEncoder()

buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))

# recombine data into a feature list and a label list
x = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(cls)
# split data
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.1)

# train a KNN classifier
model = KNeighborsClassifier(n_neighbors=9)
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print(acc)

# predict
predictions = model.predict(x_test)
names = ["unacc", "acc", "good", "vgood"]

for x in range(len(predictions)):
    print("Predictions: ", names[predictions[x]], "Data: ", x_test[x], "Actual: ", names[y_test[x]])
    # check the neighbors
    n = model.kneighbors([x_test[x]], 9, True)
    print("N: ", n)