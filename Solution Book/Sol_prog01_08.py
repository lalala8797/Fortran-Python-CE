#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 14:35:39 2023

@author: khademul
"""
# Simulate how long it takes until the game is ended by condition x or y

import numpy as np

def random_int(intl, inth):
    return np.random.randint(intl, inth + 1)

# declaration of variables and parameters
n = 2
mroll = 250
iterations = 5000
dice = np.zeros(n, dtype=int)
x = 4
y = 10
count_x = 0
max_roll = 0

# simulate dice rolls
np.random.seed(0)
for i in range(iterations):
    for j in range(mroll):
        for k in range(n):
            dice[k] = random_int(1, 6)
        if sum(dice) == x:
            count_x += 1
            max_roll = max(max_roll, j)
            break
        elif sum(dice) == y:
            max_roll = max(max_roll, j)
            break

sim_prob_x = count_x/iterations*100

# print output
print("Maximum Number of Rolls:", max_roll)
print("First condition (in %):", sim_prob_x)
print("Second condition (in %):", 100 - sim_prob_x)
