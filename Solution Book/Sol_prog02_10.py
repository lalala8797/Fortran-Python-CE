# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 12:33:17 2023

@author: u6580551
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin
from mpl_toolkits.mplot3d import Axes3D

def profit(x, p_r, p_a):
    return 3.9 - 0.1 * (p_r + p_a) + 0.05 * p_r * p_a

def plot_surface(p_r, p_a, Z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(p_r, p_a, Z)
    ax.set_xlabel('p_R')
    ax.set_ylabel('p_A')
    ax.set_zlabel('G(p_R, p_A)')
    plt.show()

# Initialize the profit matrix
p_r = np.linspace(0, 1, 4)
p_a = np.linspace(0, 1, 4)
G = np.array([[3.9 - 0.1 * (p_r[i] + p_a[j]) + 0.05 * p_r[i] * p_a[j] for j in range(4)] for i in range(4)])

# Plot the profit function using a 3D surface plot
p_r_plot, p_a_plot = np.meshgrid(p_r, p_a)
G_plot = np.array([[3.9 - 0.1 * (p_r_plot[i, j] + p_a_plot[i, j]) + 0.05 * p_r_plot[i, j] * p_a_plot[i, j] for j in range(4)] for i in range(4)])
plot_surface(p_r_plot, p_a_plot, G_plot)

# Find the maximum of the interpolated profit function
ix = np.unravel_index(np.argmax(G_plot), G_plot.shape)

# Print result to the screen
print("Interpolation:", p_r[ix[0]], p_a[ix[1]], G_plot[ix[0], ix[1]])

# Minimize the profit function using the Nelder-Mead method
x0 = np.array([0.5, 0.5])
result = fmin(lambda x: -profit(x, x0[0], x0[1]), x0)

# Print result to the screen
print("fminsearch:", result, -profit(result, x0[0], x0[1]))
