#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 11:03:47 2023

@author: khademul
"""

# Arithmetic operations with "x" and "y" which are read from the console

# Read the first number
x = float(input("Type in the 1st real number: "))

# Read the second number
y = float(input("Type in the 2nd real number: "))

# Perform arithmetic operations
addit = x + y
diffit = x - y
prodit = x * y
quotit = x / y

# Print output
print("Sum        =", addit)
print("Difference =", diffit)
print("Product    =", prodit)
print("Quotient   =", quotit)