import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score


iris_dataset = datasets.load_iris()

print(iris_dataset["data"])
print(iris_dataset['target'])
print(len(iris_dataset))
print(np.shape(iris_dataset.data))
print(np.shape(iris_dataset.target))

X = iris_dataset.data
y = iris_dataset.target
print("Class Lebels :", np.unique(y))

X_train , X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print("Label Count in y: ",np.bincount(y))
print("Label Count in y_train: ",np.bincount(y_train))
print("Label Count in y_test: ",np.bincount(y_test))

sc = StandardScaler()
sc.fit(X_train) # Estimates the parameters mean (mu) and standard deviation (sigma)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

ppn = Perceptron(eta0=0.01, random_state=42)
ppn.fit(X_train_std, y_train)
y_pred = ppn.predict(X_test_std)
print("MisClassified examples : %d" % (y_test != y_pred).sum())


print("Accuracy Score:  %.3f" % accuracy_score(y_test, y_pred))



print("Accuracy : %.3f" % ppn.score(X_test_std, y_test))




