#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 01:07:03 2023

@author: khademul
"""

import numpy as np

# Model parameters
TT = 25
gamma = 0.5
egam = 1.0 - 1.0 / gamma
beta = 0.9
alpha = 0.3
delta = 0.0
tol = 1e-5
damp = 0.25
itermax = 1000

# Model variables
g = np.zeros(TT+1)
#g = np.zeros(TT + 1)
by = np.zeros(TT + 1)
kappa = np.zeros(TT + 1)
n_p = np.zeros(TT + 1)
tax = np.zeros(TT + 1, dtype=int)
lsra_on = False
w = np.zeros(TT + 1)
r = np.zeros(TT + 1)
wn = np.zeros(TT + 1)
Rn = np.zeros(TT + 1)
p = np.zeros(TT + 1)
#p = [0.0] * (TT + 1)
tauw = np.zeros(TT + 1)
taur = np.zeros(TT + 1)
tauc = np.zeros(TT + 1)
taup = np.zeros(TT + 1)
pen = np.zeros(TT + 1)
KK = [0.0] * (TT + 1)
LL = [0.0] * (TT + 1)
YY = [0.0] * (TT + 1)
AA = [0.0] * (TT + 1)
CC = [0.0] * (TT + 1)
II = [0.0] * (TT + 1)
BB = [0.0] * (TT + 1)
GG = [0.0] * (TT + 1)
BA = [0.0] * (TT + 1)
a = np.zeros((3, TT + 1))
c = np.zeros((3, TT + 1))
util = np.zeros((3, TT + 1))
v = np.zeros(TT + 2) - 1


def initialize():
    global g, by, kappa, n_p, tax, lsra_on, LL, taup, tauc, tauw, taur, a, v, BA

    # Set baseline parameters
    g[0:3] = [0.12, 0.12, 0.0]
    by[0:TT + 1] = 0.0
    kappa[0:TT + 1] = 0.0
    n_p[0:TT + 1] = 0.2
    tax[0:TT + 1] = 1
    lsra_on = False

    # Set reform values (uncomment respective line for different tables)
    tax[1:TT + 1] = 2  # Table 6.2

    # Get labor supply and pension payments
    LL = (2.0 + n_p) / (1.0 + n_p)
    taup = kappa / ((2.0 + n_p) * (1.0 + n_p))

    # Initialize tax rates
    #tauc = 0.0
    #tauw = 0.0
    #taur = 0.0
    tauw = [0.0] * (TT + 1)
    tax = [0.0] * (TT + 1)
    taur = [0.0] * (TT + 1)


    # Initialize assets, LSRA payments, and debt holdings
    a = 0.0
    v = 0.0
    BA = [0.0] * (TT + 1)


def equilibrium():
    global TT, gamma, alpha, delta, tol, itermax, damp, tax, by, w, r, p, tauw, taur, tauc, taup, n_p, kappa, lsra_on, pen, KK, LL, YY, AA, CC, II, BB, GG
    # Run equilibrium loop
    for t in range(1, TT + 1):
        tauw[t] = tax[t] * (1 - alpha)
        taur[t] = tax[t] * alpha
        p[t] = by[t] / (2 * (1 + n_p[t]))

        # Iterate until convergence is reached
        for it in range(itermax):
            w_old = w[t]
            r_old = r[t]
            wn[t] = (1 - alpha) * ((1 - tauw[t]) * (1 - alpha) / (1 + n_p[t])) ** (-1.0 / alpha)
            Rn[t] = alpha * ((1 - taur[t]) * alpha / (1 + n_p[t])) ** ((1.0 - alpha) / alpha)
            p[t] = by[t] / (2 * (1 + n_p[t]))

            w[t] = (1 - damp) * w_old + damp * wn[t]
            r[t] = (1 - damp) * r_old + damp * Rn[t]

            if np.abs(w[t] - w_old) < tol and np.abs(r[t] - r_old) < tol:
                break

        # Compute prices, tax rates, and pension payments
        p[t] = by[t] / (2 * (1 + n_p[t]))
        tauw[t] = tax[t] * (1 - alpha)
        taur[t] = tax[t] * alpha
        tauc[t] = tax[t] * delta
        taup[t] = kappa[t] / (2 * (1 + n_p[t]))

        if lsra_on:
            pen[t] = (2 * (1 + n_p[t])) / (1 + 2 * n_p[t]) * by[t]
        else:
            pen[t] = by[t] * (1 + n_p[t])

        # Compute aggregates
        KK[t] = (alpha * r[t]) / (1 - alpha) * w[t]
        LL[t] = (1 - alpha) * r[t] / alpha * w[t]
        YY[t] = KK[t] ** alpha * LL[t] ** (1 - alpha)
        AA[t] = YY[t] / (1 - tauw[t] * (1 - alpha))
        CC[t] = (1 - alpha) * AA[t]
        II[t] = alpha * AA[t]
        BB[t] = by[t] * p[t]
        GG[t] = g[t] * p[t]
        #BA[t] = (2 * (1 + n_p[t])) / (1 + 2 * n_p[t]) * BB[t]
        BA[t] = (2 * (1 + n_p)) / (1 + 2 * n_p) * BB[t]


    return KK, LL, YY, AA, CC, II, BB, GG, BA


def main():
    initialize()
    KK, LL, YY, AA, CC, II, BB, GG, BA = equilibrium()

    # Output results
    #print("Year\tK\tL\tY\tA\tC\tI\tB\tG\tBA")
    #for t in range(TT):
        #print(f"{t}\t{KK[t]:.2f}\t{LL[t]:.2f}\t{YY[t]:.2f}\t{AA[t]:.2f}\t{CC[t]:.2f}\t{II[t]:.2f}\t{BB[t]:.2f}\t{GG[t]:.2f}\t{BA[t]:.2f}")


if __name__ == "__main__":
    main()
