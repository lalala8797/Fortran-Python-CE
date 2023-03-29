# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 11:24:02 2023

@author: u6580551
"""

import numpy as np
from scipy.optimize import minimize

mu_w = 1.0
n_w = 2
R = 1.0
beta = 1.0
gamma = 0.5
egam = 1.0 - 1.0 / gamma

def utility(x):
    a = np.zeros((3, n_w))
    a[1, 0] = x[0]
    a[2, :] = x[1:1 + n_w]

    w = np.array([0.0, 1.0])
    weight_w = np.array([0.5, 1.0 - 0.5])

    c = np.zeros((3, n_w))
    c[0, :] = mu_w - a[1, 0]
    c[1, :] = R * a[1, 0] + w - a[2, :]
    c[2, :] = R * a[2, :]
    c = np.maximum(c, 1e-10)

    utility_val = np.sum(weight_w * (c[1, :] ** egam + beta * c[2, :] ** egam))
    return -(c[0, 0] ** egam + beta * utility_val) / egam

def household2():
    bounds = [(0, mu_w)] + [(0, R * mu_w + w_val) for w_val in [0.0, 1.0]]
    x0 = [b[1] / 2 for b in bounds]

    res = minimize(utility, x0, bounds=bounds)
    x_opt = res.x

    w = np.array([0.0, 1.0])
    weight_w = np.array([0.5, 1.0 - 0.5])

    a = np.zeros((3, n_w))
    a[1, 0] = x_opt[0]
    a[2, :] = x_opt[1:1 + n_w]

    c = np.zeros((3, n_w))
    c[0, :] = mu_w - a[1, 0]
    c[1, :] = R * a[1, 0] + w - a[2, :]
    c[2, :] = R * a[2, :]

    print(" AGE   CONS   WAGE    INC    SAV")
    for j in range(3):
        print(f"{j+1:4d}{np.mean(c[j, :]):7.2f}{np.mean(w):7.2f}{np.mean(w * (j == 1) + R * a[j, :]):7.2f}{np.mean(a[j, :]):7.2f} (MEAN)")
        print(f"{'':4s}{np.std(c[j, :]):7.2f}{np.std(w):7.2f}{np.std(w * (j == 1) + R * a[j, :]):7.2f}{np.std(a[j, :]):7.2f} (STD)")

    print(f"\nE(w) = {np.sum(weight_w * w):6.2f}   Var(w) = {np.sum(weight_w * w ** 2) - np.sum(weight_w * w) ** 2:6.2f}")
    print(f"\nutility = {-res.fun:6.3f}")

if __name__ == "__main__":
    household2()
