# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 14:34:31 2023

@author: u6580551
"""

import numpy as np
from scipy.optimize import fsolve

# Global variables
Kbar = 10.0
Tbar = 30.0
alpha = 0.3
alphax = 0.4
nu = 0.5
nux = 0.5
mu = 1 - 1 / nu
mux = 1 - 1 / nux
beta = np.array([0.3, 0.6])
sigma = np.array([0.5, 0.5])
rho = 1 - 1 / sigma
a0 = np.array([0.2, 0.2])
a = np.array([[0.0, 0.3], [0.2, 0.0]])
ID = np.eye(2)
G = 3.0
tauw = 0.0
taur = 0.0
tauc = np.zeros(2)

# Market equilibrium function
def markets(x):

    global w, r, wn, rn, p, q, L, K, Y, Ybarn, Omega, PP, YD, Xd, C, ell

    # Copy producer prices and taxes
    w = 1.0
    r = x[0]
    tauc[0] = x[1]
    tauc[1] = tauc[0]

    # 1. Calculate K/Y and L/Y
    ky = a0 * ((1 - beta) + beta * ((1 - beta) * w / (beta * r))**(1 - sigma))**(-1 / rho)
    ly = a0 * (beta + (1 - beta) * (beta * r / ((1 - beta) * w))**(1 - sigma))**(-1 / rho)

    # 2. Determine producer prices
    q = w * ly + r * ky
    q = np.linalg.solve(ID.T - a, q)

    # 3. Consumer prices and demands
    p = q * (1 + tauc)
    wn = w * (1 - tauw)
    rn = r * (1 - taur)
    PP = alphax * p[0]**(1 - nux) + (1 - alphax) * p[1]**(1 - nux)
    Omega = (1 - alpha) * PP**(1 - nu) + alpha * wn**(1 - nu)
    Ybarn = wn * Tbar + rn * Kbar
    C = (1 - alpha) * Ybarn / PP**nu / Omega
    ell = alpha * Ybarn / wn**nu / Omega
    YD = Ybarn - wn * ell
    Xd = np.array([alphax * YD / p[0]**nux / PP, (1 - alphax) * YD / p[1]**nux / PP])

    # 4. Determine output levels
    Y = np.array([Xd[0] + G, Xd[1]])
    Y = np.linalg.solve(ID - a, Y)

    # 5. Compute K and L
    K = ky * Y
    L = ly * Y

    # 6. Check markets and budget
    market_check = np.array([K[0] + K[1] - Kbar,
                             q[0] * G - sum(tauc * q * Xd) - tauw * w * (Tbar - ell) - taur * r * Kbar])

    return market_check

# Initial guess
x = np.array([0.2, 0.0])

# Find market equilibrium
x_solution, _, solved, _ = fsolve(markets, x, full_output=True)

# Check whether fsolve converged
if not solved:
    print("Error in fsolve !!!")

# Get utility level
U = ((1 - alpha)**(1 / nu) * C**mu + alpha**(1 / nu) * ell**mu)**(1 / mu)

# Output
print("\nGOODS MARKET 1:")
print(f"X11 = {a[0, 0] * Y[0]:.2f}  X12 = {a[0, 1] * Y[1]:.2f}  X1 = {Xd[0]:.2f}  G = {G:.2f}  Y1 = {Y[0]:.2f}")
print(f"q1 = {q[0]:.2f}  p1 = {p[0]:.2f}  tc1 = {tauc[0]:.2f}")

print("\nGOODS MARKET 2:")
print(f"X21 = {a[1, 0] * Y[0]:.2f}  X22 = {a[1, 1] * Y[1]:.2f}  X2 = {Xd[1]:.2f}  G = {0.0:.2f}  Y2 = {Y[1]:.2f}")
print(f"q2 = {q[1]:.2f}  p2 = {p[1]:.2f}  tc2 = {tauc[1]:.2f}")

print("\nLABOR MARKET:")
print(f"L1 = {L[0]:.2f}  L2 = {L[1]:.2f}  T-ell = {Tbar - ell:.2f}")
print(f"w = {w:.2f}  wn = {wn:.2f}  tw = {tauw:.2f}")

print("\nCAPITAL MARKET:")
print(f"K1 = {K[0]:.2f}  K2 = {K[1]:.2f}  K = {Kbar:.2f}")
print(f"r = {r:.2f}  rn = {rn:.2f}  tr = {taur:.2f}")

print("\nGOVERNMENT:")
print(f"tc1 = {tauc[0] * q[0] * Xd[0]:.2f}  tc2 = {tauc[1] * q[1] * Xd[1]:.2f}  tw = {tauw * w * (Tbar - ell):.2f}  tr = {taur * r * Kbar:.2f}  G = {q[0] * G:.2f}")

print("\nUTILITY:")
print(f"U = {U:.2f}")

print("\nLeisure:")
print(f"ell = {ell:.2f}")

print("\nConsumption:")
print(f"C = {C:.2f}")

print("\nIO-TABLE:")
print(" -----------------------------------")
print(f"| {q[0] * a[0, 0] * Y[0]:.2f}  {q[0] * a[0, 1] * Y[1]:.2f} | {q[0] * Xd[0]:.2f}  {q[0] * G:.2f} | {q[0] * Y[0]:.2f}")
print(f"| {q[1] * a[1, 0] * Y[0]:.2f}  {q[1] * a[1, 1] * Y[1]:.2f} | {q[1] * Xd[1]:.2f}  {0.0:.2f} | {q[1] * Y[1]:.2f}")
print(" -----------------------------------")
print(f"| {w * L[0]:.2f}  {w * L[1]:.2f} |")
print(f"| {r * K[0]:.2f}  {r * K[1]:.2f} |")
print(" ------------------")
print(f"  {q[0] * Y[0]:.2f}  {q[1] * Y[1]:.2f}")


