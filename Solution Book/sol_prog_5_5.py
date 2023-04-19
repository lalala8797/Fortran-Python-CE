#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 22:32:16 2023

@author: khademul
"""


#need to look at the discretization function. error occurs


import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from DiscretizeTool import log_normal_discrete
from DiscretizeTool import sort

#from globals import *

mu_w = 1.0
sig_w = 0.5
n_w = 5

mu_R = 1.22
sig_R = 0.5
rho_wR = 0.0
n_R = 5

Rf = 1.0
beta = 1.0
gamma = 0.5

wR = np.zeros((n_w * n_R, 2))
weight_wR = np.zeros(n_w * n_R)

R = np.zeros(n_R)
weight_R = np.zeros(n_R)
u = np.zeros((3, n_w * n_R, n_R))

a = np.zeros((3, n_w * n_R))
omega = np.zeros((2, n_w * n_R))
c = np.zeros((3, n_w * n_R, n_R))

wag = np.zeros((3, n_w * n_R, n_R))
inc = np.zeros((3, n_w * n_R, n_R))
sav = np.zeros((3, n_w * n_R, n_R))
alp = np.zeros((3, n_w * n_R, n_R))

E_st = np.zeros(2)
Var_st = np.zeros(2)
rho_st = 0.0

def utility(x):

    global a, omega, c, u

    a[1, :] = 0.0
    a[2, :] = x[0]
    omega[0, :] = x[1]
    ic = 2
    iwR = 0
    for iw in range(n_w):
        for ir2 in range(n_R):
            a[2, iwR] = x[ic]
            omega[1, iwR] = x[ic + 1]
            ic = ic + 2
            iwR = iwR + 1

    c[0, :, :] = mu_w - a[1, 0]
    iwR = 0
    for iw in range(n_w):
        for ir2 in range(n_R):
            c[1, iwR, :] = (Rf + omega[0, 0] * (wR[iwR, 1] - Rf)) * a[1, 0] + \
                           wR[iwR, 0] - a[2, iwR]
            for ir3 in range(n_R):
                c[2, iwR, ir3] = (Rf + omega[1, iwR] * (R[ir3] - Rf)) * a[2, iwR]
            iwR = iwR + 1
    c = np.maximum(c, 1e-10)

    utility = 0.0
    iwR = 0
    for iw in range(n_w):
        for ir2 in range(n_R):
            for ir3 in range(n_R):
                prob = weight_wR[iwR] * weight_R[ir3]
                u[1, iwR, :] = 1.0 - np.exp(-c[1, iwR, 0] / gamma)
                u[2, iwR, ir3] = 1.0 - np.exp(-c[2, iwR, ir3] / gamma)
                utility += prob * (u[1, iwR, ir2] + beta * u[2, iwR, ir3])
        iwR = iwR + 1

    return -utility

def main():

    global wR, weight_wR, R, weight_R

    wR, weight_wR = log_normal_discrete(mu_w, sig_w, rho_wR, n_w, mu_R, sig_R, n_R)
    R, weight_R = sort(mu_R, sig_R, n_R)

    x0 = np.zeros(2 + 2 * n_w * n_R)
    x0[0] = 1.0
    x0[1] = 1.0
    ic = 2
    for iw in range(n_w):
        for ir in range(n_R):
            x0[ic] = 1.0
            x0[ic + 1] = 1.0
            ic = ic + 2

    bounds = [(1e-10, None)] * len(x0)

    result = minimize(utility, x0, method='L-BFGS-B', bounds=bounds, options={'ftol': 1e-10, 'maxiter': 10000})
    x = result.x

    print("Optimized parameters:", x)
    print("Utility value:", -result.fun)

    plt.plot(wR[:, 0], wR[:, 1], 'ro')
    plt.xlabel('Wealth')
    plt.ylabel('Expected returns')
    plt.show()

if __name__ == "__main__":
    main()
