#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 23:32:49 2023

@author: khademul
"""

import numpy as np

# Module globals
mu_w = 1.0
sig_w = 0.0
n_w = 5

mu_R = 1.22
sig_R = 0.0
rho_wR = 0.0
n_R = 5

Rf = 1.0
hatR = 1.0

beta = 1.0
gamma = 0.5
egam = 1 - 1 / gamma

psi = np.array([None, 0.8, 0.5])
pen = 0.0 * mu_w
xi = 0.0

wR = np.zeros((n_w * n_R, 2))
weight_wR = np.zeros(n_w * n_R)
pa = np.zeros(2)

R = np.zeros(n_R)
weight_R = np.zeros(n_R)
omega_e = np.zeros((2, n_w * n_R))
sh = np.zeros(3)

a = np.zeros((3, n_w * n_R))
c = np.zeros((3, n_w * n_R, n_R))
omega_a = np.zeros((2, n_w * n_R))

wag = np.zeros((3, n_w * n_R, n_R))
inc = np.zeros((3, n_w * n_R, n_R))
sav_b = np.zeros((3, n_w * n_R, n_R))
sav_e = np.zeros((3, n_w * n_R, n_R))
sav_a = np.zeros((3, n_w * n_R, n_R))
E_st = np.zeros(2)
Var_st = np.zeros(2)
rho_st = 0.0


def utility(x):
    global a, omega_e, omega_a, c

    # Savings
    a[0, :] = 0.0
    a[1, :] = x[0]
    omega_e[0, :] = x[1]
    omega_a[0, :] = x[2]
    ic = 3
    iwR = 0
    for iw in range(n_w):
        for ir2 in range(n_R):
            a[2, iwR] = x[ic]
            omega_e[1, iwR] = x[ic + 1]
            omega_a[1, iwR] = x[ic + 2]
            ic += 3
            iwR += 1

    # Consumption (insure consumption > 0)
    c[0, :, :] = mu_w - a[1, 0]
    iwR = 0
    for iw in range(n_w):
        for ir2 in range(n_R):
            sh[0] = (1.0 - omega_e[0, 0]) * (1.0 - omega_a[0, 0])
            sh[1] = omega_e[0, 0]
            sh[2] = (1.0 - omega_e[0, 0]) * omega_a[0, 0]
            c[1, iwR, :] = (
                (sh[0] * Rf + sh[1] * wR[iwR, 1] + sh[2] / pa[0]) * a[1, 0]
                + wR[iwR, 0]
                - a[2, iwR]
            )
            for ir3 in range(n_R):
                sh[0] = (1.0 - omega_e[1, iwR]) * (1.0 - omega_a[1, iwR]) * (1.0 - omega_a[1, iwR])
                sh[1] = omega_e[1, iwR]
                sh[2] = (1.0 - omega_e[1, iwR]) * omega_a[1, iwR]
                c[2, iwR, ir3] = (
                    (sh[0] * Rf + sh[1] * wR[iwR, 1] + sh[2] / pa[1]) * a[2, iwR]
                    + wR[iwR, 0]
                )
            iwR += 1

    # Expected utility
    EU = np.sum(np.power(c, -gamma), axis=0) * weight_R / (1.0 - gamma)

    # Certainty equivalent
    CE = np.power(np.sum(EU, axis=1) * weight_wR / egam, -egam)

    # Loss function
    loss = (
        np.sum(CE * weight_wR)
        - np.sum(EU[:, 0] * weight_wR)
        - (xi * np.sum(psi * np.sum(c, axis=2) - pen, axis=1) * weight_wR)
    )

    return loss

