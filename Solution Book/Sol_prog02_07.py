# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 12:04:37 2023

@author: u6580551
"""
# Compute the optimal tax-rate with the help of polynomial interpolation

import numpy as np
import matplotlib.pyplot as plt

# Define the data points
x = [37, 42, 45]
y = [198.875, 199.5, 196.875]

# Interpolate the data points using a polynomial function
coefficients = np.polyfit(x, y, 2)
polynomial = np.poly1d(coefficients)

# Plot the polynomial function
tax_rates = np.linspace(35, 45, 100)
tax_revenues = polynomial(tax_rates)

plt.plot(tax_rates, tax_revenues)
plt.xlabel('Tax Rate')
plt.ylabel('Tax Revenue (in billion)')
plt.title('Tax Revenue vs Tax Rate')
plt.show()

# Find the revenue maximizing tax rate
max_tax_rate = tax_rates[np.argmax(tax_revenues)]
max_tax_revenue = polynomial(max_tax_rate)

print(f'The revenue maximizing tax rate is {max_tax_rate:.2f}%')
print(f'The corresponding tax revenue is {max_tax_revenue:.2f} billion')
