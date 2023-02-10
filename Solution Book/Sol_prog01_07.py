#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 14:26:31 2023

@author: khademul
"""
# Simulate the result of rolling n dice with each k sides

import numpy as np

# declaration of parameters
n = 2
k = 6
iter = 500

# simulate dice rolls
dice = np.zeros(n)
Dsum = np.zeros(n * k)
for i in range(iter):
    for j in range(n):
        dice[j] = np.random.randint(1, k + 1)
    #Dsum[np.sum(dice) - 2] += 1
    Dsum[int(sum(dice)-2)] = Dsum[int(sum(dice)-2)] + 1.0
SimProb = Dsum / iter * 100

# print output
print('  Sum  Simulated Probability (in %)')
for i in range(n, n * k + 1):
    print('{:5d} {:30.4f}'.format(i, SimProb[i - n]))



