# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 12:28:18 2023

@author: u6580551
"""

 ## Use simplex algorithm to minimize the total cost of transportation

import numpy as np

# number of gravel pits and building sites
m = 3
n = 4

# supply and demand of gravel
supply = np.array([11, 13, 10])
demand = np.array([5, 7, 13, 6])

# transportation costs
costs = np.array([[10, 70, 100, 80], [130, 90, 120, 110], [50, 30, 80, 10]])

# initialize the initial solution
x = np.zeros((m, n))

# initialize the simplex table
table = np.zeros((m+1, n+m+1))
table[:m, :n] = costs
table[:m, n:n+m] = np.identity(m)
table[m, :n] = demand
table[:m, n+m] = supply

# find the initial solution
def pivot(table, row, col):
    table[row, :] /= table[row, col]
    for i in range(table.shape[0]):
        if i == row:
            continue
        table[i, :] -= table[i, col] * table[row, :]

while True:
    # find the entering variable
    col = np.argmin(table[m, :n])
    if table[m, col] >= 0:
        break
    
    # find the leaving variable
    ratios = table[:m, n+m]/table[:m, col]
    row = np.argmin(ratios)
    if ratios[row] <= 0:
        print("The problem is unbounded.")
        break

    # pivot
    pivot(table, row, col)

# the optimal solution
x = table[:m, n:n+m+1]
total_cost = np.sum(costs * x)
print("Optimal solution:")
print(x)
print("Total cost:", total_cost)
