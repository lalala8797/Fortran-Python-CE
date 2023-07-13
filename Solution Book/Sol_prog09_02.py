# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 12:37:00 2023

@author: u6580551
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d


# Function to create evenly spaced grid
def grid_Cons_Equi(x, x_l, x_u):
    n = len(x)
    x[0] = x_l
    x[n - 1] = x_u
    for i in range(1, n - 1):
        x[i] = x_l + (x_u - x_l) * i / (n - 1)


# Function for spline interpolation
def spline_interp(x, y, coeff):
    f = interp1d(x, y, kind='cubic')
    coeff[1:-2] = f(x[1:])
    coeff[0] = y[0]
    coeff[-2] = y[-1]



# Function for spline evaluation
def spline_eval(x, coeff, x_l, x_u):
    f = interp1d(x, coeff, kind='cubic', bounds_error=False, fill_value='extrapolate')
    return f(x)


# Function to solve for optimal consumption using fzero (root-finding)
def fzero(x_in, func):
    res = minimize_scalar(func)
    return res.x


# Function to solve for optimal consumption using fminsearch (minimization)
def fminsearch(x_in, func, x_l, x_u):
    res = minimize_scalar(func, bounds=(x_l, x_u))
    return res.x


# Module globals
beta = 0.99
alpha = 0.40
k0 = 0.1
k_l = 0.1
k_u = 0.4
sig = 1e-6
itermax = 2000
it = 0
ik = 0
iter = 0
TT = 30
c_t = np.zeros(TT + 1)
k_t = np.zeros(TT + 1)
y_t = np.zeros(TT + 1)
NK = 100
k = np.zeros(NK + 1)
c = np.zeros(NK + 1)
V = np.zeros(NK + 1)
c_new = np.zeros(NK + 1)
V_new = np.zeros(NK + 1)
coeff_c = np.zeros(NK + 3)
coeff_V = np.zeros(NK + 3)
con_lev = 0.0
x_in = 0.0
fret = 0.0
check = False
k_com = 0.0


# Function for the first-order condition
def foc(x_in):
    kplus = k_com**alpha - x_in
    cplus = spline_eval(kplus, coeff_c, k_l, k_u)
    return 1.0 / x_in - beta * alpha * kplus**(alpha - 1.0) / cplus


# Function for the utility function
def utility(x_in):
    cons = k_com**alpha - x_in
    vplus = spline_eval(x_in, k, coeff_V, k_l, k_u)
    if cons < 1e-10:
        return 1e10 * (1.0 + abs(cons))
    else:
        return -(np.log(cons) + beta * vplus)


# Function for creating output plots
def output():
    nplot = 1000
    kplot = np.linspace(k_l, k_u, nplot)
    cplot = spline_eval(kplot, k, coeff_c, k_l, k_u)

    # Plot consumption
    plt.plot(range(TT + 1), c_t, label='Numerical')
    plt.plot(range(TT + 1), c_t + k_t, label='Cons. + repl.Inv')
    plt.plot(range(TT + 1), y_t, label='Output')
    plt.xlabel('Time t')
    plt.ylabel('Goods Market')
    plt.legend()
    plt.show()

    # Plot policy function
    plt.plot(kplot, cplot)
    plt.xlabel('Current level of capital k_t')
    plt.ylabel('Policy Function c(k_t)')
    plt.show()

    # Euler equation error
    n_err = 10000
    err = 0.0
    for i in range(n_err + 1):
        k_com = k_l + (k_u - k_l) * i / n_err
        c_err = spline_eval(k_com, coeff_c, k_l, k_u)
        err_temp = abs(foc(c_err) / c_err)
        if err_temp > err:
            err = err_temp
    print('Euler equation error:', err)


# Program RamseyLog
# Start timer

# Initialize grid and policy function
grid_Cons_Equi(k, k_l, k_u)
c = k**alpha - k

# Iterate until policy function converges
for iter in range(1, itermax + 1):
    # Interpolate coefficients
    spline_interp(k, c, coeff_c)

    # Calculate decisions for every grid point
    for ik in range(NK + 1):
        # Initialize starting value and communicate resource level
        x_in = c[ik]
        k_com = k[ik]

        # Find the optimal consumption level
        x_in = fzero(x_in, foc)
        c_new[ik] = x_in

    # Get convergence level
    con_lev = np.max(np.abs(c_new - c) / np.maximum(np.abs(c), 1e-10))
    print(iter, con_lev)

    # Check for convergence
    if con_lev < sig:
        output()
        break

    c = c_new

# If no convergence with policy function iteration
if con_lev > sig:
    print('No Convergence with policy function iteration')

# Initialize value function and spline
V = np.zeros(NK + 1)
coeff_v = np.zeros(NK + 3)

# Iterate until value function converges
for iter in range(1, itermax + 1):
    # Calculate decisions for every grid point
    for ik in range(NK + 1):
        # Initialize starting value and communicate resource level
        x_in = k[ik]**alpha - c[ik]
        k_com = k[ik]

        x_in = fminsearch(x_in, utility, k_l, min(k[ik]**alpha, k_u))
        c[ik] = k[ik]**alpha - x_in
        V_new[ik] = -fret

    # Interpolate coefficients
    spline_interp(k, V_new, coeff_V)

    # Get convergence level
    con_lev = np.max(np.abs(V_new - V) / np.maximum(np.abs(V), 1e-10))
    print(iter, con_lev)

    # Check for convergence
    if con_lev < sig:
        output()
        break

    V = V_new

# If no convergence with value function iteration
if con_lev > sig:
    print('No Convergence with value function iteration')
