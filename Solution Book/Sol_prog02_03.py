#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 17:02:50 2023

@author: khademul
"""

# Compute the market equilibrium using spline interpolation

import numpy as np
import matplotlib.pyplot as plt

def minimize(a, b, tol = 1e-6):
    """Golden Section search to find the minimum of the function f(x) = x * cos(x**2)"""
    a1 = a
    b1 = b
    for iter in range(200):
        x1 = a1 + (3-np.sqrt(5))/2*(b1-a1)
        x2 = a1 + (np.sqrt(5)-1)/2*(b1-a1)
        f1 = x1*np.cos(x1**2)
        f2 = x2*np.cos(x2**2)
        if f1 < f2:
            b1 = x2
        else:
            a1 = x1
        if abs(b1-a1) < tol:
            break
    if f1 < f2:
        return x1
    else:
        return x2

def main():
    n = 4
    x_l = 0
    x_u = 5
    tol = 1e-6
    x = np.linspace(x_l, x_u, n+1)
    minimum_x = [minimize(x[i-1], x[i], tol) for i in range(1, n+1)]
    fmin = [min_x * np.cos(min_x**2) for min_x in minimum_x]
    i_global = np.argmin(fmin)
    min_global = minimum_x[i_global]
    fmin_global = fmin[i_global]
    print(f'The global minimum is located at x = {min_global:.6f}  y = {fmin_global:.6f}')
    
    x_plot = np.linspace(x_l, x_u, 100)
    y_plot = x_plot * np.cos(x_plot**2)
    plt.plot(x_plot, y_plot)
    plt.xlabel('x')
    plt.ylabel('x cos(x^2)')
    plt.show()

if __name__ == '__main__':
    main()
