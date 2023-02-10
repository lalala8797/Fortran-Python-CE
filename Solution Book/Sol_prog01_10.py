#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 14:58:53 2023

@author: khademul
"""
## Calculate utilty for different values of consumption within subroutine

n = 100
a = 0.5
b = 5.0
u = [0.0] * n

def utility_int(a, b, u):
    for i in range(n):
        u[i] = (a + (b - a) * i / (n - 1)) ** 2

utility_int(a, b, u)

for i in range(n):
    print("{:.5f}".format(u[i]))