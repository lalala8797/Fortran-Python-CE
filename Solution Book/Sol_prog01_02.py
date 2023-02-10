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