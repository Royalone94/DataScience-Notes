# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

labels = ['a', 'b', 'c']
my_data = [10,20,30]
arr = np.array(my_data)
d = {'a':10, 'b':20, 'c':30}
ser = pd.Series(data = my_data, index=labels)
ser1 = pd.Series([1,2,3,4], ['USA', 'Germany', 'USSR', 'Japan'])
ser2 = pd.Series([1,2,5,4], ['USA', 'Germany', 'Italy', 'Japan'])
A = ser1['USA']
B = ser1 + ser2