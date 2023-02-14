# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 11:38:20 2023

@author: u6580551
"""
 ## Compute the integal for of exp(-x) and |x|^0.5 numerically
import numpy as np
from scipy.integrate import quad

def f1(x):
    return np.exp(-x)

def f2(x):
    return np.sqrt(np.abs(x))

def trapezoid_rule(f, a, b, n):
    h = (b-a)/n
    integral = 0.5 * (f(a) + f(b))
    for i in range(1, n):
        integral += f(a + i*h)
    integral *= h
    return integral

def simpson_rule(f, a, b, n):
    h = (b-a)/n
    integral = f(a) + f(b)
    for i in range(1, n, 2):
        integral += 4 * f(a + i*h)
    for i in range(2, n-1, 2):
        integral += 2 * f(a + i*h)
    integral *= h/3
    return integral

def gauss_legendre(f, a, b, n):
    x, w = np.polynomial.legendre.leggauss(n)
    x = (b-a)*x/2 + (b+a)/2
    return np.dot(f(x), w)*(b-a)/2

a = -1
b = 1
n = [10, 20, 30]

f = [f1, f2]
integral_true = [quad(f1, a, b)[0], quad(f2, a, b)[0]]

for i in range(2):
    print(f"\nFunction f{i+1}:")
    print(f"True integral: {integral_true[i]}")
    for j in range(3):
        integral_approx = [trapezoid_rule(f[i], a, b, n[j]),
                           simpson_rule(f[i], a, b, n[j]),
                           gauss_legendre(f[i], a, b, n[j])]
        print(f"\nNumber of nodes: {n[j]}")
        print(f"Trapezoid rule approximation: {integral_approx[0]}")
        print(f"Simpson's rule approximation: {integral_approx[1]}")
        print(f"Gauss-Legendre approximation: {integral_approx[2]}")
