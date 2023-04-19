#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 23:16:42 2023

@author: khademul
"""

#need to check descritization tool


import numpy as np
from scipy.optimize import minimize
from DiscretizeTool import log_normal_discrete, NormalDiscrete1

# Global constants
psi = np.array([0.8, 0.5])
mu_w = 1.0
sig_w = 0.9
n_w = 5

R = 1.0
beta = 1.0
gamma = 0.5
egam = 1.0 - 1.0 / gamma
nu = 0.05
pen = mu_w

# Global arrays
w = np.zeros(n_w)
weight_w = np.zeros(n_w)
p_a = np.zeros(2)

a = np.zeros((4, n_w))
omega = np.zeros((2, n_w))
c = np.zeros((3, n_w))

wag = np.zeros((3, n_w))
inc = np.zeros((3, n_w))
sav = np.zeros((3, n_w))
alp = np.zeros((3, n_w))
b = np.zeros((3, n_w))


# Utility function of the household
def utility(x):
    global a, omega, c, b

    # Savings
    a[0, :] = 0.0
    a[1, :] = x[0]
    omega[0, :] = x[1]
    ic = 2
    for iw in range(n_w):
        a[2, iw] = x[ic]
        omega[1, iw] = x[ic + 1]
        a[3, iw] = x[ic + 2]
        ic += 3

    # Consumption (ensure consumption > 0)
    c[0, :] = mu_w - a[1, 0]
    c[1, :] = R * (1.0 - omega[0, 0]) * a[1, 0] + omega[0, 0] * a[1, 0] / p_a[0] + w - a[2, :]
    c[2, :] = R * (1.0 - omega[1, :]) * a[2, :] + omega[1, :] * a[2, :] / p_a[1] + omega[0, 0] * a[1, 0] / p_a[0] + pen - a[3, :]
    c = np.maximum(c, 1e-10)

    # Bequests (ensure bequests > 0)
    b[0, :] = R * (1.0 - omega[0, 0]) * a[1, 0]
    b[1, :] = R * (1.0 - omega[1, :]) * a[2, :]
    b[2, :] = R * a[3, :]
    b = np.maximum(b, 1e-10)

    # Expected utility of period 3
    utility_val = 0.0
    for iw in range(n_w):
        utility_val += weight_w[iw] * beta**2 * psi[0] * psi[1] * (c[2, iw]**egam + beta * nu * b[2, iw]**egam)

    # Expected utility of period 2
    for iw in range(n_w):
        utility_val += weight_w[iw] * beta * psi[0] * (c[1, iw]**egam + beta * (1.0 - psi[1]) * nu * b[1, iw]**egam)


    # Add first period utility
    utility_val = -(c[0, 0]**egam + beta * (1.0 - psi[0]) * nu * b[0, 0]**egam + utility_val) / egam

    return utility_val


# Calculate expected value
def E(x):
    return np.sum(x * weight_w)


# Calculate standard deviation
def Std(x):
    E_x = E(x)
    return np.sqrt(max(np.sum(x**2 * weight_w) - E_x**2, 0))


# Main program
if __name__ == "__main__":
    

    # Discretize log(wage)
    w, weight_w = log_normal_discrete(mu_w, sig_w, n_w)

    # Calculate annuity factors
    p_a[0] = psi[0] / R + psi[0] * psi[1] / R**2
    p_a[1] = psi[1] / R

    # Lower and upper border and initial guess
    low = np.zeros(3 * n_w + 2)
    up = np.zeros(3 * n_w + 2)
    up[0] = mu_w
    up[1] = 1.0
    ic = 2
    for iw in range(n_w):
        up[ic] = R * mu_w + w[iw]
        up[ic + 1] = 1.0
        up[ic + 2] = R * (R * mu_w + w[iw]) + pen
        ic += 3
    x = np.full(3 * n_w + 2, 0.2)

    # Minimization routine
    res = minimize(utility, x, bounds=list(zip(low, up)), tol=1e-14)
    x_opt = res.x

    # Set up data for output
    for iw in range(n_w):
        wag[0, iw] = mu_w
        inc[0, iw] = mu_w
        sav[0, iw] = a[1, 0] * (1.0 - omega[0, 0])
        alp[0, iw] = a[1, 0] * omega[0, 0]

        wag[1, iw] = w[iw]
        inc[1, iw] = w[iw] + R * (1.0 - omega[0, 0]) * a[1, 0] + omega[0, 0] * a[1, 0] / p_a[0]
        sav[1, iw] = a[2, iw] * (1.0 - omega[1, iw])
        alp[1, iw] = a[2, iw] * omega[1, iw]

        wag[2, iw] = 0.0
        inc[2, iw] = R * (1.0 - omega[1, iw]) * a[2, iw] + omega[1, iw] * a[2, iw] / p_a[1] + omega[0, 0] * a[1, 0] / p_a[0] + pen
        sav[2, iw] = a[3, iw]
        alp[2, iw] = 0.0

    # Output
    print("AGE   CONS   WAGE    INC   SREG   SANN")
    for j in range(3):
        print("{:4d}{:7.2f}{:7.2f}{:7.2f}{:7.2f}{:7.2f} (MEAN)".format(
            j + 1, E(c[j]), E(wag[j]), E(inc[j]), E(sav[j]), E(alp[j])))
        print("    {:7.2f}{:7.2f}{:7.2f}{:7.2f}{:7.2f} (STD)".format(
            Std(c[j]), Std(wag[j]), Std(inc[j]), Std(sav[j]), Std(alp[j])))

    print("\nE(w) = {:6.2f}   Var(w) = {:6.2f}".format(
        np.sum(weight_w * w), np.sum(weight_w * w**2) - np.sum(weight_w * w)**2))

    print("omega1 = {:6.2f}".format(omega[0, 0]))
    print("omega2 = {:6.2f}".format(E(omega[1])))

    print("\nutil   = {:7.3f}".format(-res.fun))

