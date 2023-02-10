#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 13:09:30 2023

@author: khademul
"""
# Evaluate german tax code for the income read from the console

def TaxFunction():
    # declaration of variables
    y = None
    x = None
    z = None
    T = None
    average = None
    marginal = None

    # read the income from console
    y = float(input('Type in the income: '))

    x = (y - 8130) / 10000
    z = (y - 13469) / 10000

    if y < 8131:
        T = 0
        average = 0
        marginal = 0
    elif y < 13470:
        T = (933.70 * x + 1400) * x
        average = T / y
        marginal = (1867.4 * x + 1400) / 10000
    elif y < 52882:
        T = (228.74 * z + 2397) * z + 1014
        average = T / y
        marginal = (457.48 * z + 2397) / 10000
    elif y < 250731:
        T = 0.42 * y - 8196
        average = T / y
        marginal = 0.42
    else:
        T = 0.45 * y - 15718
        average = T / y
        marginal = 0.45

    # print output
    print("tax burden:       ", T)
    print("average tax:      ", average * 100)
    print("marginal tax rate:", marginal * 100)

TaxFunction()
