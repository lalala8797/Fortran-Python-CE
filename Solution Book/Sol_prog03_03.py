# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 12:17:36 2023

@author: u6580551
"""

from scipy.optimize import fsolve
import numpy as np

Kbar = 10.0
Tbar = 30.0
alpha = 0.3
alphax = 0.5
nu = 0.5
nux = 0.5
mu = 1.0 - 1.0 / nu
mux = 1.0 - 1.0 / nux
beta = np.array([0.3, 0.6])

Ybar = 0.0
Omega = 0.0
PP = 0.0
w = 0.0
r = 0.0
p = np.zeros(2)
L = np.zeros(2)
K = np.zeros(2)
Y = np.zeros(2)
U = 0.0
YD = 0.0
C = 0.0
ell = 0.0

def markets(x):
    global p, w, r, Ybar, Omega, PP, U, YD, C, ell, Y, L, K

    p[0] = 1.0
    p[1] = x[0]
    w = x[1]
    r = x[2]

    Ybar = w * Tbar + r * Kbar
    PP = alphax * p[0]**(1 - nux) + (1 - alphax) * p[1]**(1 - nux)
    Omega = (1 - alpha) * PP**(1 - nu) + alpha * w**(1 - nu)

    eq1 = 1.0 / p[0] - (beta[0] / w)**beta[0] * ((1.0 - beta[0]) / r)**(1.0 - beta[0])
    eq2 = 1.0 / p[1] - (beta[1] / w)**beta[1] * ((1.0 - beta[1]) / r)**(1.0 - beta[1])
    eq3 = ((beta[0] * alphax / p[0]**(nux - 1.0) + beta[1] * (1.0 - alphax) / p[1]**(nux - 1.0)) * 
           (w**(nu - 1.0) * Omega - alpha) / (w**nu * Omega * PP) * Ybar + alpha * Ybar / w**nu / Omega - Tbar)

    return [eq1, eq2, eq3]

x0 = np.array([0.3, 0.3, 0.3])
x, check = fsolve(markets, x0, full_output=True)[:2]

if check != 1:
    print('Error in fzero !!!')

C = (1.0 - alpha) * Ybar / PP**nu / Omega
ell = alpha * Ybar / (w**nu * Omega)
YD = Ybar - w * ell
Y[0] = alphax * YD / p[0]**nux / PP
Y[1] = (1.0 - alphax) * YD / p[1]**nux / PP
L = beta * p * Y / w
K = (1.0 - beta) * p * Y / r
U = ((1.0 - alpha)**(1.0 / nu) * C**mu + alpha**(1.0 / nu) * ell**mu)**(1.0 / mu)

# Output
print('Goods Market 1 :')
print('X1 = ', Y[0], 'Y1 = ', Y[0])
print('Goods Market 2 :')
print('X2 = ', Y[1], 'Y2 = ', Y[1])
print('Labour Market :')
print('L1 = ', L[0], 'L2 = ', L[1], 'L= ', Tbar-ell, 'w= ',w)
print('Capital Market :')
print('K1 = ', K[0], 'K2 = ', K[1], 'K = ', Kbar, 'r= ', r)
