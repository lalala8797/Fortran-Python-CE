# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 11:11:14 2023

@author: u6580551
"""

import numpy as np
from scipy.optimize import minimize


class Globals:
    w = 1.0
    R = 1.0
    beta = 1.0
    theta = 0.5
    nu = 0.5
    gamma = 0.5
    egam = 1.0 - 1.0 / gamma
    delh = 1.0
    a_low = 0.5

    @staticmethod
    def utility(x):
        a = np.zeros(3)
        a[0] = 0
        ah = x[0]
        a[1] = x[1]
        a[2] = x[2]

        p = Globals.R - 1 + Globals.delh

        c = np.zeros(3)
        c[0] = Globals.w - a[1] - ah
        c[1] = Globals.R * a[1] + Globals.w - a[2] - Globals.delh * ah
        c[2] = Globals.R * a[2] + (1 - p - Globals.delh) * ah
        c = np.maximum(c, 1e-10)
        ah = max(ah, 1e-10)

        u = np.zeros(3)
        u[0] = (Globals.theta * c[0]**Globals.nu + (1 - Globals.theta) * ah**Globals.nu)**(Globals.egam / Globals.nu) / Globals.egam
        u[1] = (Globals.theta * c[1]**Globals.nu + (1 - Globals.theta) * ah**Globals.nu)**(Globals.egam / Globals.nu) / Globals.egam
        u[2] = (Globals.theta * c[2]**Globals.nu + (1 - Globals.theta) * ah**Globals.nu)**(Globals.egam / Globals.nu) / Globals.egam

        utility_value = -(u[0] + Globals.beta * u[1] + Globals.beta**2 * u[2])

        return utility_value


def housing():
    x = np.zeros(3)
    low = np.array([0, -Globals.a_low, -Globals.a_low])
    up = np.array([Globals.w + Globals.a_low, Globals.w, Globals.R * Globals.w + Globals.w])
    x = up / 3

    res = minimize(Globals.utility, x, bounds=[(low[i], up[i]) for i in range(len(x))])

    x_opt = res.x
    fret = -res.fun
    c = np.zeros(3)
    c[0] = Globals.w - x_opt[1] - x_opt[0]
    c[1] = Globals.R * x_opt[1] + Globals.w - x_opt[2] - Globals.delh * x_opt[0]
    c[2] = Globals.R * x_opt[2] + (1 - Globals.R + Globals.delh) * x_opt[0]

    ah = max(x_opt[0], 1e-10)

    print(" AGE   CONS   DCONS  WAGE    INC    SAV   UTIL")
    print(f"{1:4d}{c[0]:7.2f}{ah:7.2f}{Globals.w:7.2f}{Globals.w:7.2f}{x_opt[1]:7.2f}")
    print(f"{2:4d}{c[1]:7.2f}{ah:7.2f}{Globals.w:7.2f}{Globals.w + Globals.R * x_opt[1] - Globals.delh * ah:7.2f}{x_opt[2]:7.2f}")
    print(f"{3:4d}{c[2]:7.2f}{ah:7.2f}{0.00:7.2f}{Globals.R * x_opt[2] + (1 - Globals.delh + Globals.R) * ah:7.2f}{0.00:7.2f}{fret:7.2f}")


if __name__ == "__main__":
    housing()






