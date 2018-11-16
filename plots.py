import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def myplot(xminus, yminus, xplus, yplus, title, xlim = 5, ylim = 5):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    axes = plt.gca()
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    axes.set_xlim([-xlim,ylim])
    axes.set_ylim([-xlim,ylim])
    plt.xlabel("X_0")
    plt.ylabel("X_1")
    plt.scatter(xminus, yminus, c='r', cmap=cm_bright)
    plt.scatter(xplus, yplus, c='b', cmap=cm_bright)
    ax.legend(["y = - 1", "y = + 1"])
    plt.savefig("{}.svg".format(title))
    plt.show()

# Mean vector and covariance matrix
muplus = np.array([0., 0.])
muminus = np.array([0., 0.])

SigmaMinus = np.array([[ 1. , 0.], [0. ,  1.]])
SigmaPlus = np.array([[ 2. , 0.], [0. ,  0.5]])


xminus, yminus = np.random.multivariate_normal(muminus, SigmaMinus, 1000).T
xplus, yplus = np.random.multivariate_normal(muplus, SigmaPlus, 1000).T
myplot(xminus, yminus, xplus, yplus,"default", xlim = 5, ylim = 5)

#Covariance matrix is bigger
SigmaPlusGreater = np.array([[ 4. , 0.], [0. ,  2.]])
xplus, yplus = np.random.multivariate_normal(muplus, SigmaPlusGreater, 1000).T
myplot(xminus, yminus, xplus, yplus, "biggercov", xlim = 5, ylim = 5)

#Covariance matrix is smaller
SigmaPlusSmaller = np.array([[ 0.5 , 0.], [0. ,  0.125]])
xplus, yplus = np.random.multivariate_normal(muplus, SigmaPlusSmaller, 1000).T
myplot(xminus, yminus, xplus, yplus, "smallercov", xlim = 5, ylim = 5)

#Mean is different
muplus = np.array([1., 1.])
muminus = np.array([-1., -1.])
xminus, yminus = np.random.multivariate_normal(muminus, SigmaMinus, 1000).T
xplus, yplus = np.random.multivariate_normal(muplus, SigmaPlus, 1000).T
myplot(xminus, yminus, xplus, yplus, "meandiff", xlim = 5, ylim = 5)