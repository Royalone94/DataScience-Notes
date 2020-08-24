# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from numpy.random import randn

A = np.random.seed(101)
df = pd.DataFrame(randn(5,4), ['A', 'B', 'C', 'D', 'E'], ['W','X','Y','Z'])
B = type(df['W'])
C = type(df)
D = df['W']
E = df[['W', 'Z']]
df['new'] = df['W'] + df['Y']
df.drop('new', axis=1, inplace=True)
df1 = df.drop('E')
F = df.loc['A']
G = df.iloc[2]
H = df.loc['B', 'Y']
I = df.loc[['A', 'B'],['W', 'Y']]

J = df > 0
K = df[J]
L = df[df['Z'] < 0]
M = df[df['W']>0]
N = df[df['W']>0]['X']

O = df[(df['W']>0) & (df['Y']>1)]
P = df[(df['W']>0) | (df['Y']>1)]

newind = 'CA NY WY OR CO'.split()
df['States'] = newind
df2 = df.set_index('States')
