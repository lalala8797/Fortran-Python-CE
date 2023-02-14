#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 16:38:26 2023

@author: khademul

"""
# Solve the intertemporal household optimization problem

import scipy.optimize as spopt

def foc(x_in):
    return [x_in[0]**(-1/gamma) - x_in[2], beta*x_in[1]**(-1/gamma) - x_in[2]/(1+r), w - x_in[0] - x_in[1]/(1+r)]

def utility(x_in):
    return -(x_in**egam/egam + beta*((w-x_in)*(1+r))**egam/egam)

gamma = 0.5
egam = 1 - 1/gamma
beta = 1
r = 0
w = 1

x_root = spopt.fsolve(foc, [0.1, 0.1, 0.1])
print("Result with fsolve:")
print("------------------")
print("c_1    : ", x_root[0])
print("c_2    : ", x_root[1])
print("lambda : ", x_root[2])

res = spopt.fminbound(lambda x: -utility(x), 0, w)
print("Result with fminbound:")
print("---------------------")
print("c_1    : ", res)
print("c_2    : ", (w - res)*(1+r))
