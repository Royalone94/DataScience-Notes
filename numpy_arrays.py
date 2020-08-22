# -*- coding: utf-8 -*-
#numpy arrays
my_list = [1,2,3]
import numpy as np
arr = np.array(my_list)
my_mat = [[1,2,3],[4,5,6],[7,8,9]]
my_mat_np = np.array(my_mat)
A = np.arange(0,11,2)
B = np.linspace(0,5,10)
C = np.eye(4)
D = np.random.rand(5, 5)
E = np.random.randn(2, 4)
F = np.random.randint(100)
G = np.random.randint(0, 50, 10)
H = arr.reshape(3,1)
I = my_mat_np.reshape(3,3)
J = I.min()
K = I.max()
L = I.argmax()
M = I.reshape(1,9)
N = I.dtype

