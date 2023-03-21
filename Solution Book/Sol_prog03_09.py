# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 10:17:47 2023

@author: u6580551
"""

import numpy as np

# Global parameters
Kbar = 10.0
Tbar = 30.0
alpha = np.array([0.3, 0.4])
beta = np.array([0.3, 0.6])
a0 = np.array([0.2, 0.2])
G = 3.0
ID = np.array([[1, 0], [0, 1]], dtype=float)
a = np.array([[0, 0.3], [0.2, 0]], dtype=float)
tauw = 0.0
taur = 0.0
tau = np.array([0.0, 0.0])

# Market equilibrium function
def markets(x):
    global w, r, tau, L, K, Y, ly, ky, Xd, T, ell, p, q, U, lmbda

    # Copy producer prices and taxes
    w = 1.0
    r = x[0]
    tau[0] = x[1]
    tau[1] = 0.0
    lmbda = 1

    # 1. Calculate K/Y and L/Y
    ky = a0 * ((1 - beta) / beta * w / r) ** beta
    ly = a0 * (beta / (1 - beta) * r / w) ** (1 - beta)

    # 2. Determine producer prices
    q = w * ly + r * ky
    a[0, 1] = a[0, 1] * (1 + lmbda * tau[0])
    q = np.linalg.solve(ID.T - a, q)

    # 3. Consumer prices and demands
    p = q * (1 + tau)
    wn = w * (1 - tauw)
    rn = r * (1 - taur)
    Ybarn = wn * Tbar + rn * Kbar
    Xd = alpha / p * Ybarn
    ell = (1 - alpha[0] - alpha[1]) / wn * Ybarn

    # 4. Determine output levels
    Y = np.array([Xd[0] + G, Xd[1]])
    a[0, 1] = a[0, 1] / (1 + lmbda * tau[0])
    Y = np.linalg.solve(ID - a, Y)

    # 5. Compute K and L
    K = ky * Y
    L = ly * Y

    # 6. Compute company (sectoral) tax revenues
    T = np.array([
        tau[0] * q[0] * Y[0] - np.sum(tau * q * a[:, 0]) * Y[0],
        tau[1] * q[1] * Y[1] - (1 - lmbda) * np.sum(tau * q * a[:, 1]) * Y[1]
    ])

    # 7. Check markets and budget
    markets_result = np.array([
        K[0] + K[1] - Kbar,
        p[0] * G - np.sum(T) - tauw * w * (Tbar - ell) - taur * r * Kbar
    ])

    return markets_result

# Main program
x = np.array([1.8, 0.3])
lambda_value = 1

from scipy.optimize import fsolve
x_sol, _, ier, _ = fsolve(markets, x, full_output=True)

if ier != 1:
    print("Error in fsolve !!!")
    exit()

# Get utility level
U = Xd[0] ** alpha[0] * Xd[1] ** alpha[1] * ell ** (1 - alpha[0] - alpha[1])

# Output results
print("GOODS MARKET 1:")
print(f"X11 = {a[0, 0] * Y[0]:.2f}  X12 = {a[0, 1] * Y[1]:.2f}  X1 = {Xd[0]:.2f}  G = {G:.2f}  Y1 = {Y[0]:.2f}")
print(f"q1 = {q[0]:.2f}  p1 = {p[0]:.2f}  t1 = {tau[0]:.2f}")

print("\nGOODS MARKET 2:")
print(f"X21 = {a[1, 0] * Y[0]:.2f}  X22 = {a[1, 1] * Y[1]:.2f}  X2 = {Xd[1]:.2f}  G = {0.0:.2f}  Y2 = {Y[1]:.2f}")
print(f"q2 = {q[1]:.2f}  p2 = {p[1]:.2f}  t2 = {tau[1]:.2f}")

print("\nLABOR MARKET:")
print(f"L1 = {L[0]:.2f}  L2 = {L[1]:.2f}  T-l = {Tbar - ell:.2f}  Diff = {Tbar - ell - np.sum(L):.2f}")
#print(f"w = {w:.2f}  wn = {wn:.2f}  tw = {tauw:.2f}")

print("\nCAPITAL MARKET:")
print(f"K1 = {K[0]:.2f}  K2 = {K[1]:.2f}  K = {Kbar:.2f}  Diff = {Kbar - np.sum(K):.2f}")
#print(f"r = {r:.2f}  rn = {rn:.2f}  tr = {taur:.2f}")

print("\nGOVERNMENT:")
print(f"t1 = {T[0]:.2f}  t2 = {T[1]:.2f}  tw = {tauw * w * (Tbar - ell):.2f}  tr = {taur * r * Kbar:.2f}  G = {q[0] * G:.2f}")

print("\nUTILITY:")
print(f"U = {U:.2f}\n")

# IO-Table
print("IO-TABLE:")
print("-----------------------------------")
print(f"| {q[0] * a[0, 0] * Y[0]:.2f}  {q[0] * a[0, 1] * Y[1]:.2f} | {q[0] * Xd[0]:.2f}  {q[0] * G:.2f} | {q[0] * Y[0]:.2f}")
#print(f"| {q[1] * a[1, 0] * Y[0]:.2f}  {q[1] * a[1, 1] * Y

