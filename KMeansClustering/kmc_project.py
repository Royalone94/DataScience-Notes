# -*- coding: utf-8 -*-

# K Means Clustering Project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('College_Data', index_col=0)
head = df.head()
describe = df.describe()

#sns.lmplot(x='Room.Board', y='Grad.Rate', data=df, hue='Private', fit_reg=False, palette='coolwarm', size=6, aspect=1)
#sns.lmplot(x='Outstate', y='F.Undergrad', data=df, hue='Private', fit_reg=False, palette='coolwarm', size=6, aspect=1)
#g = sns.FacetGrid(df, hue='Private', palette='coolwarm', size=6, aspect=2)
#g = g.map(plt.hist, 'Outstate', bins=20, alpha=0.7)
g = sns.FacetGrid(df, hue='Private', palette='coolwarm', size=6, aspect=2)
g = g.map(plt.hist, 'Grad.Rate', bins=20, alpha=0.7)

df[df['Grad.Rate']>100]
df['Grad.Rate']['Cazenovia College'] = 100
df[df['Grad.Rate']>100]
g = sns.FacetGrid(df, hue='Private', palette='coolwarm', size=6, aspect=2)
g = g.map(plt.hist, 'Grad.Rate', bins=20, alpha=0.7)


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2)
kmeans.fit(df.drop('Private', axis=1))
kmeans.cluster_centers_

def converter(private):
    if private == 'Yes':
        return 1
    else:
        return 0
    
df['Cluster'] = df['Private'].apply(converter)
head2 = df.head()

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(df['Cluster'], kmeans.labels_))
print(classification_report(df['Cluster'], kmeans.labels_))