import numpy as np
import math
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


x = np.random.uniform(-4,4,1000)
y = [math.sin(x_) + 0.5 * math.sin(3*x_) + np.random.normal(0, math.sqrt(0.01)) for x_ in x]

plt.scatter(x,y)
plt.show()