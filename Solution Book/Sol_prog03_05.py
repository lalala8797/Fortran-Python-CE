# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 11:34:50 2023

@author: u6580551
"""

import numpy as np
from scipy.optimize import fsolve

# Global variables
Kbar = 10.0
Tbar = 30.0
alpha = 0.3
alphax = 0.5
nu = 0.5
nux = 0.5
mu = 1.0 - 1.0 / nu
mux = 1.0 - 1.0 / nux
beta = np.array([0.3, 0.6])
sigma = np.array([0.5, 0.5])
rho = 1.0 - 1.0 / sigma
G = 3.0
tauw = 0.0
taur = 0.0
tauc = np.zeros(2)

L = np.zeros(2)
K = np.zeros(2)
Y = np.zeros(2)
ly = np.zeros(2)
ky = np.zeros(2)
Ybarn = Omega = PP = w = wn = r = rn = 0
p = np.zeros(2)
q = np.zeros(2)
U = YD = C = ell = 0
Xd = np.zeros(2)


def markets(x):
    global w, r, tauw, taur, tauc, ly, ky, q, p, wn, rn, PP, Omega
    global Ybarn, C, ell, YD, Xd, Y, L, K

    w = 1.0
    r = x[0]
    tauw = -x[1]
    tauc[0] = x[1]
    tauc[1] = tauc[0]
    ly = (beta + (1.0 - beta) * (beta * r / ((1.0 - beta) * w))**(1.0 - sigma))**(-1.0 / rho)
    ky = ((1.0 - beta) + beta * ((1.0 - beta) * w / (beta * r))**(1.0 - sigma))**(-1.0 / rho)
    q = ly * w + ky * r
    p = q * (1.0 + tauc)
    wn = w * (1.0 - tauw)
    rn = r * (1.0 - taur)
    PP = alphax * p[0]**(1.0 - nux) + (1.0 - alphax) * p[1]**(1.0 - nux)
    Omega = (1.0 - alpha) * PP**(1.0 - nu) + alpha * wn**(1.0 - nu)

    Ybarn = wn * Tbar + rn * Kbar
    C = (1.0 - alpha) * Ybarn / PP**nu / Omega
    ell = alpha * Ybarn / wn**nu / Omega
    YD = Ybarn - wn * ell
    Xd[0] = alphax * YD / p[0]**nux / PP
    Xd[1] = (1.0 - alphax) * YD / p[1]**nux / PP
    Y[0] = Xd[0] + G
    Y[1] = Xd[1]
    L = ly * Y
    K = ky * Y

    return np.array([K.sum() - Kbar, q[0] * G - tauc.dot(q * Xd) - tauw * w * (Tbar - ell)    - taur * r * Kbar])

# Initial guess
x_initial = np.array([0.5, 0.4])

# Find market equilibrium
x_solution, infodict, ier, message = fsolve(markets, x_initial, full_output=True)

if ier != 1:
    print("Error in fsolve:", message)
else:
    # Get utility level
    U = ((1.0 - alpha)**(1.0 / nu) * C**mu + alpha**(1.0 / nu) * ell**mu)**(1.0 / mu)

    # Output
    print("\nGOODS MARKET 1 :")
    print(f"X1 = {Y[0]:.2f}  Y1 = {Y[0]:.2f}  q1 = {p[0]:.2f}  p1 = {p[0]:.2f}")

    print("\nGOODS MARKET 2 :")
    print(f"X2 = {Y[1]:.2f}  Y2 = {Y[1]:.2f}  q2 = {p[1]:.2f}  p2 = {p[1]:.2f}")

    print("\nLABOR MARKET :")
    print(f"L1 = {L[0]:.2f}  L2 = {L[1]:.2f}  L  = {Tbar - ell:.2f}  w  = {w:.2f}")

    print("\nCAPITAL MARKET :")
    print(f"K1 = {K[0]:.2f}  K2 = {K[1]:.2f}  K  = {Kbar:.2f}  r  = {r:.2f}")

    print("\nGOVERNMENT :")
    print(f"tc1 = {tauc[0] * q[0] * Xd[0]:.2f}  tc2 = {tauc[1] * q[1] * Xd[1]:.2f}  tw = {tauw * w * (Tbar - ell):.2f}  tr = {taur * r * Kbar:.2f}  G  = {q[0] * G:.2f}")

    print("\nUTILITY :")
    print(f"U  = {U:.2f}")

