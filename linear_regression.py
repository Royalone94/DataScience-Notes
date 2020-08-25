# -*- coding: utf-8 -*-

# Linear Regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('USA_Housing.csv')
#print(df.head)
#print(df.info())
#print(df.describe())

#sns.pairplot(df)
#sns.distplot(df['Price'])
#sns.heatmap(df.corr())

#print(df.columns)
df.dropna(inplace=True)
X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population', 'Price', 'Address']]
y = df['Price']
df.dropna(inplace=True)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
mm = X_train.columns

cdf = pd.DataFrame(lm.coef, X.columns, columns=['Coeff'])