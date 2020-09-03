# -*- coding: utf-8 -*-

# Support Vector Machines Project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

iris = sns.load_dataset('iris')

# sns.pairplot(iris, hue='species', palette='Dark2')
setosa = iris[iris['species'] == 'setosa']
# sns.kdeplot(setosa['sepal_width'], setosa['sepal_length'], cmap='plasma', shade=True, shade_lowest=False)

from sklearn.model_selection import train_test_split, GridSearchCV
X = iris.drop('species', axis=1)
y = iris['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

from sklearn.svm import SVC
svc_model = SVC()
svc_model.fit(X_train, y_train)

predictions = svc_model.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
param_grid = {'C':[0.1, 1, 10, 100], 'gamma':[1, 0.1, 0.01, 0.001]}
grid = GridSearchCV(SVC(), param_grid, verbose=2)
grid.fit(X_train, y_train)
grid_predictions = grid.predict(X_test)
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))