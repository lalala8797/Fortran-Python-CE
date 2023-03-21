# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 10:48:52 2023

@author: u6580551
"""
#there is error in the code

import numpy as np
from scipy.optimize import root

# Constants
alpha = np.array([0.3, 0.4])
a0 = np.array([0.2, 0.2])
a = np.array([[0.0, 0.3], [0.2, 0.0]])
G = 3.0
ID = np.identity(2)
Tbar = np.array([30.0, 10.0])
Kbar = np.array([10.0, 30.0])
beta = np.array([[0.3, 0.6], [0.3, 0.6]])
tauw = np.array([0.0, 0.0])
taur = np.array([0.0, 0.0])
tauc = np.zeros((2, 2))
r = 0.0
lambda_ = 0

# Variables
Ybarn = np.zeros(2)
q = np.zeros((2, 2))
p = np.zeros((2, 2))
w = np.zeros(2)
wn = np.zeros(2)
rn = np.zeros(2)
T = np.zeros((2, 2))
wh = np.zeros((2, 2))
Xd = np.zeros((2, 2))
Y = np.zeros((2, 2))
ky = np.zeros((2, 2))
ly = np.zeros((2, 2))
K = np.zeros((2, 2))
L = np.zeros((2, 2))
U = np.zeros(2)
ell = np.zeros(2)


def markets(x):
    global w, r, tauc
    global Ybarn, q, p, wn, rn, Xd, ell, ky, ly, Y, K, L, T, wh

    # Copy producer prices and taxes
    w[0] = 1.0
    w[1] = x[0]
    r = x[1]
    tauc[0, :] = x[2:4]
    tauc[1, :] = tauc[0, :]

    # 1. Calculate K/Y and L/Y
    for i in range(2):
        ky[:, i] = a0 * ((1 - beta[:, i]) / beta[:, i] * w[i] / r)**beta[:, i]
        ly[:, i] = a0 * (beta[:, i] / (1 - beta[:, i]) * r / w[i])**(1 - beta[:, i])

    # 2. Determine producer prices
    q[:, 0] = (w[0] * ly[:, 0] + r * ky[:, 0]) * (1 + lambda_ * tauc[:, 0])
    q[:, 0] = np.linalg.solve(ID - a.T, q[:, 0])
    q[0, 1] = (a[1, 0] * q[1, 0] + (w[1] * ly[0, 1] + r * ky[0, 1]) * (1 + lambda_ * tauc[0, 1])) / (1 - a[0, 0])
    q[1, 1] = q[1, 0]

    # 3. Consumer prices and demands
    for i in range(2):
        p[:, i] = q[:, i] * (1 + (1 - lambda_) * tauc[:, i])
        wn[i] = w[i] * (1 - tauw[i])
        rn[i] = r * (1 - taur[i])

    # 4. Household optimization
    for i in range(2):
        ell[i] = (wn[i] * Tbar[i] * rn[i] * Kbar[i]) / (wn[i] * Tbar[i] + rn[i] * Kbar[i])
        Xd[i, i] = ell[i] / Tbar[i]
        Xd[i, 1 - i] = (1 - ell[i]) / Tbar[1 - i]

        # 5. Output
    Y = np.dot(a, q)
    for i in range(2):
        Y[i, i] += (w[i] * ly[:, i] + r * ky[:, i]).sum()

    # 6. Capital and labor demands
    K = ky * Y
    L = ly * Y

    # 7. Utility
    U = np.sum(np.log(np.sum(Xd * p, axis=1)))

    # 8. Government revenue
    T = np.sum(np.sum(a * q * tauc, axis=1))
    wh = np.sum(a * q * tauc, axis=1)

    # Equilibrium conditions
    eq1 = np.sum(K, axis=1) - Kbar
    eq2 = np.sum(L, axis=1) - Tbar
    eq3 = np.sum(wh, axis=0) - G

    return np.hstack([eq1, eq2, eq3])



# Solving the system of equations
initial_guess = np.array([1.0, 0.1, 0.0, 0.0,0.0])
solution = root(markets, initial_guess)
x_solution = solution.x

# Print results
print("Solution found:")
print(f"w = {x_solution[0]:.4f}, r = {x_solution[1]:.4f}, tauc = [{x_solution[2]:.4f}, {x_solution[3]:.4f}]")

# Compute other variables
markets(x_solution)
print("\nOutput (Y):")
print(Y)
print("\nCapital (K):")
print(K)
print("\nLabor (L):")
print(L)
print("\nUtility (U):")
print(U)
print("\nGovernment revenue (T):")
print(T)
print("\nHousehold demands (Xd):")
print(Xd)

