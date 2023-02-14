#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 17:31:48 2023

@author: khademul
"""

import numpy as np
from scipy.optimize import minimize

def utility(x_in):
    c = np.zeros(3)
    c[0] = w + w/(1.0+r) - x_in[0]/(1.0+r) - x_in[1]/(1.0+r)**2.0
    c[1] = x_in[0]
    c[2] = x_in[1]
    return -np.sum(np.log(c))

w = 100.0
r = 0.05

a = np.array([0.0, 0.0])
b = np.array([w + w*(1.0+r), w*(1.0+r) + w*(1.0+r)**2.0])
x_in = np.array([w/2.0, w/2.0])

res = minimize(utility, x_in, bounds=list(zip(a,b)))

c = np.zeros(3)
c[0] = w + w/(1.0+r) - res.x[0]/(1.0+r) - res.x[1]/(1.0+r)**2.0
c[1] = res.x[0]
c[2] = res.x[1]

print("Result with minimize:")
print("----------------------")
print("c_1    : ", c[0])
print("c_2    : ", c[1])
print("c_3    : ", c[2])
