#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 10:02:22 2023

@author: khademul
"""
import numpy as np
from scipy.optimize import minimize

# Constants
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
egam = 1.0 - 1.0 / gamma
F = 0.04

# Global variables
wR = np.zeros((n_w * n_R, 2))
weight_wR = np.zeros(n_w * n_R)

R = np.zeros(n_R)
weight_R = np.zeros(n_R)

a = np.zeros((3, n_w * n_R))
omega = np.zeros((2, n_w * n_R))
c = np.zeros((3, n_w * n_R, n_R))

# Utility function of the household who invests in stocks
def utility_st(x):
    global a, omega, c
    
    # Savings
    a[0, :] = 0.0
    a[1, :] = x[0]
    omega[0, :] = x[1]
    ic = 2
    iwR = 0
    for iw in range(n_w):
        for ir2 in range(n_R):
            a[2, iwR] = x[ic]
            omega[1, iwR] = x[ic + 1]
            ic += 2
            iwR += 1

    # Consumption (insure consumption > 0)
    c[0, :, :] = mu_w - a[1, 0] - F
    iwR = 0
    for iw in range(n_w):
        for ir2 in range(n_R):
            c[1, iwR, :] = (Rf + omega[0, 0] * (wR[iwR, 1] - Rf)) * a[1, 0] + wR[iwR, 0] - a[2, iwR]
            for ir3 in range(n_R):
                c[2, iwR, ir3] = (Rf + omega[1, iwR] * (R[ir3] - Rf)) * a[2, iwR]
            iwR += 1
    c = np.maximum(c, 1e-10)

    # Expected utility of periods 2 and 3
    utility_st = 0.0
    iwR = 0
    for iw in range(n_w):
        for ir2 in range(n_R):
            for ir3 in range(n_R):
                prob = weight_wR[iwR] * weight_R[ir3]
                utility_st += prob * (c[1, iwR, 0] ** egam + beta * c[2, iwR, ir3] ** egam)
            iwR += 1

    # Utility function
    utility_st = -(c[0, 0, 0] ** egam + beta * utility_st) / egam

    return utility_st

p

# Utility function of the household who invests in bonds
"""
def utility_b(x):
    global a, c
    
    # Savings
    a[0, :] = 0.0
    a[1, :] = x[0]
    ic = 1
    iwR = 0
    for iw in range(n_w):
        for ir2
"""