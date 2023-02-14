#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 15:03:54 2023

@author: khademul
"""
# Perform lu-decomposition and solve linear equation system

import numpy as np
from scipy.linalg import lu

A = np.array([[1, 5, 2, 3], [1, 6, 8, 6], [1, 6, 11, 2], [1, 7, 17, 4]])
b = np.array([1, 2, 1, 1])

print("A:")
print(A)

P, L, U = lu(A)
A_test = np.dot(L, U)

print("L:")
print(L)
print("U:")
print(U)
print("A_test:")
print(A_test)
print("-----------------------------")

x = np.linalg.solve(A, b)
b_test = np.dot(A, x)

print("x:")
print(x)
print("b_test:")
print(b_test)
