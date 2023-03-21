# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 14:46:59 2023

@author: u6580551
"""

import numpy as np

# Global variables and parameters
Kbar = 10.0
Tbar = 30.0
alpha = np.array([0.3, 0.4])
beta = np.array([0.3, 0.6])
a0 = np.array([0.2, 0.2])
a = np.array([[0.0, 0.3], [0.2, 0.0]])
ID = np.array([[1.0, 0.0], [0.0, 1.0]])
G = 15.0
tauw = 0.0
taur = 0.0
tauc = np.zeros(2)

L = np.zeros(2)
K = np.zeros(2)
Y = np.zeros(2)
ly = np.zeros(2)
ky = np.zeros(2)
Xd = np.zeros(2)

Ybarn = None
w = None
wn = None
r = None
rn = None
p = np.zeros(2)
q = np.zeros(2)
U = None
ell = None


def markets(x):
    global w, tauc, r, q, wn, rn, Ybarn, Xd, ell, Y, L, K

    # Copy producer prices and taxes
    r = 1.0
    q[1] = 1.0
    w = x[0]
    tauc[0] = x[1]
    tauc[1] = tauc[0]

    # 1. Calculate K/Y and L/Y
    ky = a0 * ((1.0 - beta) / beta * w / r) ** beta
    ly = a0 * (beta / (1.0 - beta) * r / w) ** (1.0 - beta)

    # 2. Determine producer prices
    q[0] = (a[1, 0] * q[1] + w * ly[0] + r * ky[0]) / (1.0 - a[0, 0])

    # 3. Consumer prices and demands
    p = q * (1.0 + tauc)
    wn = w * (1.0 - tauw)
    rn = r * (1.0 - taur)
    Ybarn = wn * Tbar + rn * Kbar
    Xd = alpha / p * Ybarn
    ell = (1.0 - alpha[0] - alpha[1]) / wn * Ybarn

    # 4. Determine output levels
    Y[0] = Xd[0] + G
    Y[1] = Tbar - ell
    mat = np.array([[1.0 - a[0, 0], -a[0, 1]], [ly[0], ly[1]]])
    Y = np.linalg.solve(mat, Y)

    # 5. Compute K and L
    K = ky * Y
    L = ly * Y

    # 6. Check markets and budget
    market_vals = np.zeros(2)
    market_vals[0] = (1.0 - a[1, 1]) * q[1] - a[0, 1] * q[0] - w * ly[1] - r * ky[1]
    market_vals[1] = q[0] * G - np.sum(tauc * q * Xd) - tauw * w * (Tbar - ell) - taur * r * Kbar

    return market_vals


def fzero(x, func):
    from scipy.optimize import root

    sol = root(func, x)
    return sol.x, sol.success


if __name__ == "__main__":
    # Initial guess
    x = np.array([0.2, 0.0])

    # Find market equilibrium
    x, check = fzero(x, markets)

    # Check whether fzero converged
    if not check:
        print("Error in fzero !!!")
        exit()

    # Get utility level
    U = Xd[0] ** alpha[0] * Xd[1] ** alpha[1] * ell ** (1.0 - alpha[0] - alpha[1])

    # Output results
    # (Note: The output formatting is simplified compared to the original Fortran code)
    print("\nGOODS MARKET 1:")
    print("X11=", a[0, 0] * Y[0], "X12=", a[0, 1] * Y[1], "X1 =", Xd[0], "G  =", G, "Y1 =", Y[0])
    print("q1 =", q[0], "p1 =", p[0], "tc1=", tauc[0])

    print("\nGOODS MARKET 2:")
    print("X21=", a[1, 0] * Y[0], "X22=", a[1, 1] * Y[1], "X2 =", Xd[1], "G  =", 0.0, "Y2 =", Y[1], "EXP=", Y[1] - Xd[1] - a[1, 0] * Y[0] - a[1, 1] * Y[1])
    print("q2 =", q[1], "p2 =", p[1], "tc2=", tauc[1])

    print("\nLABOR MARKET:")
    print("L1 =", L[0], "L2 =", L[1], "T-ell=", Tbar - ell)
    print("w  =", w, "wn =", wn, "tw =", tauw)

    print("\nCAPITAL MARKET:")
    print("K1 =", K[0], "K2 =", K[1], "K  =", Kbar, "EXP=", Kbar - K[0] - K[1])
    print("r  =", r, "rn =", rn, "tr =", taur)

    print("\nGOVERNMENT:")
    print("tc1=", tauc[0] * q[0] * Xd[0], "tc2=", tauc[1] * q[1] * Xd[1], "tw =", tauw * w * (Tbar - ell), "tr =", taur * r * Kbar, "G  =", q[0] * G)

    print("\nUTILITY:")
    print("U  =", U)
