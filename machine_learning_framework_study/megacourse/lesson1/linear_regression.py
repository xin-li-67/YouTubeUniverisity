# supervised learning algorithm
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use("ggplot")

####################
# prepare the data #
####################
data = pd.read_csv("./datasets/student/student-mat.csv", sep=";")
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

# define what need to be predicted:
# use studytime, failures, absences, G1, and G2 to predict G3
predict = "G3"

# return a new data without G3
x = np.array(data.drop([predict], 1))
# return a new data with only G3
y = np.array(data[predict])
# split 10% data into test dataset
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

# train multiple times for best score
best_score = 0
for _ in range(20):
    # design algorithm #
    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print("Accuracy: " + str(acc))

    if acc > best_score:
        best_score = acc
        # save a modle
        with open("./models/studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)

# open a modle
pickle_in = open("./models/studentmodel.pickle", "rb")
# load the model
linear = pickle.load(pickle_in)

# view the slope value and intercept constants
print("-------------------------")
print('Coefficient: ', linear.coef_)
print('Intercept: ', linear.intercept_)
print("-------------------------")


# results on specifit students
predictions = linear.predict(x_test)
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

# plot the model
pivot = "G1"
plt.scatter(data[pivot], data["G3"])
plt.legend(loc=4)
plt.xlabel(pivot)
plt.ylabel("Final Grade")
plt.show()