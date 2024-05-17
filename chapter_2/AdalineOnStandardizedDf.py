import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from Adaline import AdalineGD

# Plotting Decision boundaries for 2-D datasets
from matplotlib.colors import ListedColormap

data_src = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

dataframe = pd.read_csv(data_src, header=None, encoding='utf-8')
y = dataframe.iloc[:100,4].values # Select setosa and Versicolor data
y = np.where(y == "Iris-setosa", 0,1)
X = dataframe.iloc[:100, [0,2]].values # Extract sepal length and petal length


def plot_decision_regions(X,y, classifier, resolution = 0.02):
    # Setup marker generator and color map
    markers = ('o', 's', '^','<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() -1, X[:,0].max() + 1
    x2_min, x2_max = X[:, 1].min() -1, X[:,1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    plt.contourf(xx1, xx2, lab, alpha = 0.3, cmap= cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # Plot class Examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x = X[y==cl, 0],
                    y = X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    label = f"class {cl}",
                    edgecolors='black')
        

    # plt.xlabel('Sepal Length [cm]')
    # plt.ylabel('Petal Length [cm]')
    # plt.legend()
    # plt.show()

# # Plotting without standarized dataset
# plot_decision_regions(X, y, classifier=ada1)
# plt.title("Adaline -  Gradient Descent")
# plt.xlabel('Sepal Length')
# plt.ylabel("Petal Length")
# plt.legend(loc = 'upper left')
# plt.tight_layout()
# plt.show()


# Standarizing dataset
X_std = np.copy(X)
X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()    # Standardarization : (x-mean)/std
X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()    # Standardarization : (x-mean)/std
ada_gd = AdalineGD(n_iter=50, lr=0.1)
ada_gd.fit(X_std, y)
plot_decision_regions(X_std, y, classifier=ada_gd)
plt.title("Adaline -  Gradient Descent")
plt.xlabel('Standarized Sepal Length')
plt.ylabel("Standarized Petal Length")
plt.legend(loc = 'upper left')
plt.tight_layout()
plt.show()
plt.plot(range(1, len(ada_gd.losses) + 1),ada_gd.losses , marker = 'o')
plt.xlabel("Epochs")
plt.ylabel("Mean Squared Error")
plt.tight_layout()
plt.show()