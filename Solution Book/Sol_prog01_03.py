#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 11:18:15 2023

@author: khademul
"""

rvar1 = [10**9, 10**9, 10**9]
rvar2 = [None, 10**10, 10**10]
x = [0.000000000003, 0.000000000003]
y = [3.1415926535, 3.1415926535]

print(" Without d0          With base d0          With expo d0")
print("Exp  9", rvar1[0], rvar1[1], rvar1[2])
print("Exp 10", rvar2[0], rvar2[1], rvar2[2])
print("Precision   ", x[0], x[1])
print("Precision     ", y[0], y[1])

"""In Python, all floating-point numbers are stored as float data type, 
which provides high precision by default. The d0 suffix used in Fortran to 
indicate double precision is not needed in Python, as all floating-point 
numbers are stored as float.

The code performs the same calculation as in the original Fortran program and 
prints the results to the console. Note that the format of the output has been 
simplified in the Python code, as the f22.2 and f30.25 formats are not 
available in Python."""