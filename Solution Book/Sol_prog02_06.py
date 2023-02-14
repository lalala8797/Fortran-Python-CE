# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 11:47:07 2023

@author: u6580551
"""
## Compute the the change in relative consumer surplus

import numpy as np
from scipy.integrate import fixed_quad

def price(d):
    return 4 / (d + 1) ** 2

def consumer_surplus(p, d_range):
    integral, _ = fixed_quad(lambda d: (price(d) - p) * d, d_range[0], d_range[1], n=10)
    return integral

d_range = [0, 10]
p1 = 3
p2 = 1
consumer_surplus1 = consumer_surplus(p1, d_range)
consumer_surplus2 = consumer_surplus(p2, d_range)
relative_change = (consumer_surplus2 - consumer_surplus1) / consumer_surplus1
print("Relative change in consumer surplus:", relative_change)
