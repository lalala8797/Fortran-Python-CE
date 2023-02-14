# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 12:07:53 2023

@author: u6580551
"""

# Interpolate Cosine Function linearly and with a spline

import numpy as np
import matplotlib.pyplot as plt

# Set parameters
n = 10
n_plot = 1000
pi = 3.1415926535

# Get nodes and data for plotting
x_plot = np.linspace(0, 2*pi, n_plot+1)
y_real = np.cos(x_plot)

# Get nodes and data for interpolation
x = np.linspace(0, 2*pi, n+1)
y = np.cos(x)

# Compute slope and intercept for each subinterval
m = (y[1:] - y[:-1])/(x[1:] - x[:-1])
t = y[1:] - m*x[1:]

# Piecewise linear interpolation by hand
y_plot_1 = np.zeros(n_plot+1)
for j in range(n_plot+1):
    i = min(int((x_plot[j] - x[0])/(x[-1] - x[0])*n), n-1)
    y_plot_1[j] = m[i]*x_plot[j] + t[i]

# Piecewise linear interpolation using NumPy
y_plot_2 = np.interp(x_plot, x, y)

# Cubic spline interpolation using SciPy
from scipy.interpolate import CubicSpline
cs = CubicSpline(x, y)
y_plot_3 = cs(x_plot)

# Print output
print("Approximation Error")
print("-------------------")
print("Piecewise Linear (by hand): ", np.max(np.abs(y_plot_1 - y_real)))
print("Piecewise Linear (NumPy): ", np.max(np.abs(y_plot_2 - y_real)))
print("Cubic Spline: ", np.max(np.abs(y_plot_3 - y_real)))

# Plot the results
plt.plot(x_plot, y_plot_1, label='Piecewise Linear (by hand)')
plt.plot(x_plot, y_plot_2, label='Piecewise Linear (NumPy)')
plt.plot(x_plot, y_plot_3, label='Cubic Spline')
plt.plot(x_plot, y_real, label='Cos(x)')
plt.xlabel('x')
plt.ylabel('Cos(x)')
plt.legend()
plt.show()
