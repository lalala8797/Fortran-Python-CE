#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 13:21:30 2023

@author: khademul
"""
"""
This program calculates the n-th element of the Fibonacci series using two methods: an iterative solution and a closed-form solution (Binet's formula). The program first reads the value of n from the user, then it calculates and displays the n-th element of the Fibonacci series using both methods.

The iterative solution calculates the n-th element by defining the first two elements and then iteratively updating the values of the current and previous elements in a loop.

The closed-form solution uses Binet's formula, which calculates the n-th element of the Fibonacci series as a function of the golden ratio.

Finally, the program plots the difference between the two methods for the first 100 elements of the Fibonacci series, using a plotting tool (here, the "toolbox" module is used). The difference is calculated as the absolute value of the difference between the iterative solution and Binet's formula.
"""
def fibonacci(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        a, b = 0, 1
        for i in range(2, n + 1):
            a, b = b, a + b
        return b

def binet_formula(n):
    import math
    golden_ratio = (1 + math.sqrt(5)) / 2
    return int((golden_ratio**n - (1 - golden_ratio)**n) / math.sqrt(5))

def plot_difference(n):
    import matplotlib.pyplot as plt
    x = list(range(1, n + 1))
    y = [abs(fibonacci(i) - binet_formula(i)) for i in x]
    plt.plot(x, y)
    plt.xlabel("n")
    plt.ylabel("Difference")
    plt.title("Difference between iterative solution and Binet's formula")
    plt.show()

# read input
n = int(input("Type in the requested element of the series: "))

# print output
print(f"{n}-element of the Fibonacci series (with iterative solution): {fibonacci(n)}")
print(f"{n}-element of the Fibonacci series (with Binet's formula): {binet_formula(n)}")

plot_difference(n)
