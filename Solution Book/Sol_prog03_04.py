# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 12:31:48 2023

@author: u6580551
"""

import numpy as np
from scipy.optimize import fsolve

# define global parameters
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

# define variables
L = np.zeros(2)
K = np.zeros(2)
Y = np.zeros(2)
ly = np.zeros(2)
ky = np.zeros(2)
p = np.zeros(2)
Ybar = 0.0
Omega = 0.0
PP = 0.0
w = 0.0
r = 0.0
YD = 0.0
C = 0.0
ell = 0.0
U = 0.0

# define function to determine market equilibrium
def markets(x):
    global L, K, Y, ly, ky, p, Ybar, Omega, PP, w, r, YD, C, ell

    # calculate prices
    w = 1.0
    r = x
    ly = (beta + (1.0 - beta) * (beta * r / ((1.0 - beta) * w)) ** (1.0 - sigma)) ** (-1.0 / rho)
    ky = ((1.0 - beta) + beta * ((1.0 - beta) * w / (beta * r)) ** (1.0 - sigma)) ** (-1.0 / rho)
    p = ly * w + ky * r
    PP = alphax * p[0] ** (1.0 - nux) + (1.0 - alphax) * p[1] ** (1.0 - nux)
    Omega = (1.0 - alpha) * PP ** (1.0 - nu) + alpha * w ** (1.0 - nu)

    # calculate other economic variables
    Ybar = w * Tbar + r * Kbar
    C = (1.0 - alpha) * Ybar / PP ** nu / Omega
    ell = alpha * Ybar / w ** nu / Omega
    YD = Ybar - w * ell
    Y[0] = alphax * YD / p[0] ** nux / PP
    Y[1] = (1.0 - alphax) * YD / p[1] ** nux / PP
    L = ly * Y
    K = ky * Y

    # get market equations
    return K[0] + K[1] - Kbar

# initial guess
x0 = 0.5

# find market equilibrium
x = fsolve(markets, x0)[0]

# get utility level
U = ((1.0 - alpha) ** (1.0 / nu) * C ** mu + alpha ** (1.0 / nu) * ell ** mu) ** (1.0 / mu)

# Output
print('Goods Market 1 :')
print('X1 = ', Y[0], 'Y1 = ', Y[0])
print('Goods Market 2 :')
print('X2 = ', Y[1], 'Y2 = ', Y[1])
print('Labour Market :')
print('L1 = ', L[0], 'L2 = ', L[1], 'L= ', Tbar-ell, 'w= ',w)
print('Capital Market :')
print('K1 = ', K[0], 'K2 = ', K[1], 'K = ', Kbar, 'r= ', r)
