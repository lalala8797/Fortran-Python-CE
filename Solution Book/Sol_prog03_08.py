# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 14:53:41 2023

@author: u6580551
"""

import numpy as np
from scipy.optimize import root
import scipy

# Set parameters
Kbar = 10.0
Tbar = 30.0
alpha = np.array([0.3, 0.4])
beta = np.array([0.3, 0.6])
a0 = np.array([0.2, 0.2])
a = np.array([[0.0, 0.3], [0.2, 0.0]])
ID = np.eye(2)
G = 3.0
tauk = np.zeros(2)
taul = np.array([0.0, 0.2])
tauw = 0.0
taur = 0.0
tauc = np.zeros(2)

def lu_solve(A, b):
    P, L, U = scipy.linalg.lu(A)
    y = scipy.linalg.solve_triangular(L, P @ b, lower=True)
    x = scipy.linalg.solve_triangular(U, y)
    return x

def markets(x):
    global w, r, tauc, wg, rg, p, q, L, K, Y, ly, ky, Xd, Ybarn, wn, rn, ell
    
    # Copy producer prices and taxes
    w = 1.0
    r = x[0]
    tauc[0] = x[1]
    tauc[1] = tauc[0]
    rg = (1.0 + tauk) * r
    wg = (1.0 + taul) * w

    # Calculate K/Y and L/Y
    ky = a0 * (((1.0 - beta) * wg) / (beta * rg))**beta
    ly = a0 * ((beta * rg) / ((1.0 - beta) * wg))**(1.0 - beta)

    # Determine producer prices
    q = wg * ly + rg * ky
    q = lu_solve(ID - np.transpose(a), q)

    # Consumer prices and demands
    p = q * (1.0 + tauc)
    wn = w * (1.0 - tauw)
    rn = r * (1.0 - taur)
    Ybarn = wn * Tbar + rn * Kbar
    Xd = alpha / p * Ybarn
    ell = (1.0 - alpha[0] - alpha[1]) / wn * Ybarn

    # Determine output levels
    Y = np.array([Xd[0] + G, Xd[1]])
    Y = lu_solve(ID - a, Y)

    # Compute K and L
    K = ky * Y
    L = ly * Y

    # Check markets and budget
    market_vals = np.zeros(2)
    market_vals[0] = K[0] + K[1] - Kbar
    market_vals[1] = q[0] * G - np.sum(tauc * q * Xd) - tauw * w * (Tbar - ell) - taur * r * Kbar

    return market_vals

# Initial guess
x = np.array([0.2, 0.0])

# Find market equilibrium
sol = root(markets, x)
x = sol.x
check = sol.success

# Check whether fzero converged
if not check:
    print("Error in fzero !!!")
    exit()

# Get utility level
U = Xd[0] ** alpha[0] * Xd[1] ** alpha[1] * ell ** (1.0 - alpha[0] - alpha[1])

#Output
print("GOODS MARKET 1:")
print(f"X11 = {a[0, 0] * Y[0]:.2f} X12 = {a[0, 1] * Y[1]:.2f} X1 = {Xd[0]:.2f} G = {G:.2f} Y1 = {Y[0]:.2f}")
print(f"q1 = {q[0]:.2f} p1 = {p[0]:.2f} tc1 = {tauc[0]:.2f} tl = {taul[0]:.2f} tk = {tauk[0]:.2f}")

print("\nGOODS MARKET 2:")
print(f"X21 = {a[1, 0] * Y[0]:.2f} X22 = {a[1, 1] * Y[1]:.2f} X2 = {Xd[1]:.2f} G = {0:.2f} Y2 = {Y[1]:.2f}")
print(f"q2 = {q[1]:.2f} p2 = {p[1]:.2f} tc2 = {tauc[1]:.2f} tl = {taul[1]:.2f} tk = {tauk[1]:.2f}")

print("\nLABOR MARKET:")
print(f"L1 = {L[0]:.2f} L2 = {L[1]:.2f} T-l = {Tbar - ell:.2f}")
print(f"w = {w:.2f} wn = {wn:.2f} tw = {tauw:.2f} DIFF = {L[0] + L[1] + ell - Tbar:.2f}")
