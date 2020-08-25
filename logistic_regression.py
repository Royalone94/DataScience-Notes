# -*- coding: utf-8 -*-

# Logistic Regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('titanic_train.csv')
#print(train.head())
#print(train.isnull())
#sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')

sns.set_style('whitegrid')

#sns.countplot(x='Survived', hue='Sex', data=train, palette='RdBu_r')

#sns.countplot(x='Survived', hue='Pclass', data=train)

#sns.distplot(train['Age'].dropna(), kde=False, bins=30)

#train['Age'].plot.hist(bins=35)

#train.info()

#sns.countplot(x='SibSp', data=train)

#train['Fare'].hist(bins=40, figsize=(10,4))

#plt.figure(figsize=(10,7))
#sns.boxplot(x='Pclass', y='Age', data=train)

def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age
    
train['Age'] = train[['Age', 'Pclass']].apply(impute_age, axis=1)
#sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')
train.drop('Cabin', axis=1, inplace=True)
#sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')
train.dropna(inplace=True)
#sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')

df_sex_male = pd.get_dummies(train['Sex'], drop_first=True)
embark = pd.get_dummies(train['Embarked'], drop_first=True)
train = pd.concat([train, df_sex_male, embark], axis=1)
train.drop(['Sex', 'Embarked', 'Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)


X = train.drop('Survived', axis=1)
y = train['Survived']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression(max_iter=7600)
logmodel.fit(X_train, y_train)
predictions = logmodel.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))

print(confusion_matrix(y_test, predictions))














