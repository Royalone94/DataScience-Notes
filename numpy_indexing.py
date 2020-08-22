# -*- coding: utf-8 -*-

import numpy as np
arr = np.arange(0,11)
A = arr[8]
B = arr[1:5]
C = arr[0:5]
D = arr[:6]
E = arr[5:]
slice_of_arr = arr[0:6]
slice_of_arr[:] = 99
arr_copy = arr.copy()
arr_copy[:] = 100
arr_2d = np.array([[5,10,15],[20,25,30],[35,40,45]])
F = arr_2d[0]
G = arr_2d[1][1]
H = arr_2d[1, 2]
I = arr_2d[:2, 1:]
J = arr_2d[:2]

arr2 = np.arange(1,11)
bool_arr = arr2 > 5
bool_arr2 = arr2[bool_arr]
K = arr2[arr2<3]
