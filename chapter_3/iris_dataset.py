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

# Plot decision regions 
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt




def plot_decision_regions(X, y, classifier, test_index=None, resolution = 0.02):

    # 3 Setup marker , generator, and Colormap
    markers = ('o', 's','^', 'v', ">")
    colors = ('red','blue', 'lightgreen', "grey", "cyan")
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:,0].min()-1, X[:,0].max()+1
    x2_min, x2_max = X[:,1].min()-1, X[:,1].max()+1
    # x3_min, x3_max = X[:,2].min()-1, X[:,2].max()+1
    # x4_min, x4_max = X[:,3].min()-1, X[:,3].max()+1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                            np.arange(x2_min, x2_max, resolution),
                            # np.arange(x3_min, x3_max, resolution),
                            # np.arange(x4_min, x4_max, resolution)
                            )
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    #plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x = X[y == cl, 0],
                    y = X[y == cl, 1],
                    alpha=0.8,
                    c = colors[idx],
                    marker= markers[idx],
                    label = f'Class {cl}',
                    edgecolors='black')
    # highlight test examples
    if test_index:
        # plot all examples
        X_test, y_test = X[test_index, :], y[test_index]
        plt.scatter(X_test[:, 0], X_test[:, 1],
        c = "none", 
        edgecolors="black",
        alpha=1.0,
        linewidth = 1,
        marker='o',
        s=100,
        label="Test set")


X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X=X_combined_std,
                    y=y_combined,
                    classifier=ppn,
                    test_index=range(105,150))

plt.xlabel("Petal Length [standardarized]")
plt.ylabel("Petal Width [standarized]")
plt.legend(loc= 'upper left')
plt.tight_layout()
plt.show()




