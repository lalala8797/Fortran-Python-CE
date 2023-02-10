#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 14:41:29 2023

@author: khademul
"""
# Calculate utiltiy for different values of consumption and gamma

import matplotlib.pyplot as plt

def utility(c, gamma):
    return c**(1-gamma) / (1-gamma)

c_read = float(input("Type in a consumption level: "))

gamma_array = [0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]

for ig in range(len(gamma_array)):
    print("c = {:.2f} U = {:.2f} gamma = {:.1f}".format(c_read, utility(c_read, gamma_array[ig]), gamma_array[ig]))

NC = 1000
c = [float(ic)/NC for ic in range(NC+1)]

for ig in range(len(gamma_array)):
    u = [utility(c[ic], gamma_array[ig]) for ic in range(NC+1)]

    label = "gamma = {:.4f}".format(gamma_array[ig])
    plt.plot(c, u, label=label)

plt.legend()
plt.show()
