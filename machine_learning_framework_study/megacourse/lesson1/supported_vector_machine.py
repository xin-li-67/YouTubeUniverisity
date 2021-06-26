# supervised learning algorithm
import sklearn

from sklearn import svm
from sklearn import metrics
from sklearn import datasets

cancer = datasets.load_breast_cancer()

# print(cancer.feature_names)
# print(cancer.target_names)

x = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

# implement a svm and adding a kernel (a kernel is just a function like f(x1, x2) -> x3),
# which could make the original dataset from like 2d to 3d so that we could have a hyper-plane
clf = svm.SVC(kernel="linear")
clf.fit(x_train, y_train)

# predict
y_pred = clf.predict(x_test)

acc = metrics.accuracy_score(y_test, y_pred)

print(acc)