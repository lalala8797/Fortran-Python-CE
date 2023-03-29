# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 11:17:45 2023

@author: u6580551
"""

import numpy as np
from scipy.optimize import minimize

class Globals:
    mu = 0.5
    sigma = 0.5
    w = 1.0
    R = 1.0
    beta = 1.0
    gamma = 0.5
    egam = 1.0 - 1.0 / gamma
    NH = 20

def utility(x):
    a = np.zeros(3)
    a[1] = x[0]
    a[2] = x[1]

    c = np.zeros(2)
    c[0] = Globals.w - a[1]
    c[1] = Globals.R * a[1] + Globals.w - a[2]

    hc = np.linspace(Globals.mu - Globals.sigma, Globals.mu + Globals.sigma, Globals.NH)
    c3 = Globals.R * a[2] - hc
    c = np.maximum(c, 1e-10)
    c3 = np.maximum(c3, 1e-10)

    weight_h = np.ones(Globals.NH) / Globals.NH
    expected_utility = np.sum(weight_h * c3 ** Globals.egam)

    return -(c[0] ** Globals.egam + Globals.beta * c[1] ** Globals.egam + Globals.beta ** 2 * expected_utility) / Globals.egam

def health_expend():
    bounds = [(0, Globals.w), (0, Globals.R * Globals.w + Globals.w)]
    x0 = [0.5 * Globals.w, 0.5 * (Globals.R * Globals.w + Globals.w)]

    res = minimize(utility, x0, bounds=bounds)
    x_opt = res.x

    c = np.zeros(2)
    c[0] = Globals.w - x_opt[0]
    c[1] = Globals.R * x_opt[0] + Globals.w - x_opt[1]

    hc = np.linspace(Globals.mu - Globals.sigma, Globals.mu + Globals.sigma, Globals.NH)
    c3 = Globals.R * x_opt[1] - hc
    weight_h = np.ones(Globals.NH) / Globals.NH

    print(" AGE   CONS   WAGE    INC    SAV   EH")
    print(f"{1:4d}{c[0]:7.2f}{Globals.w:7.2f}{Globals.w:7.2f}{x_opt[0]:7.2f}")
    print(f"{2:4d}{c[1]:7.2f}{Globals.w:7.2f}{Globals.w + Globals.R * x_opt[0]:7.2f}{x_opt[1]:7.2f}")
    print(f"{3:4d}{np.mean(c3):7.2f}{0.00:7.2f}{Globals.R * x_opt[1]:7.2f}{0.00:7.2f}{np.mean(hc):7.2f} (MEAN)")
    print(f"{'':4s}{np.std(c3):7.2f}{0.00:7.2f}{0.00:7.2f}{0.00:7.2f}{np.std(hc):7.2f} (STD)")

if __name__ == "__main__":
    health_expend()
