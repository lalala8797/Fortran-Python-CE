import numpy as np
from scipy.optimize import root_scalar


# Define global parameters
Kbar = 10.0
Lbar = 20.0
alpha = 0.3
nu = 0.5
mu = 1 - 1 / nu
beta = np.array([0.3, 0.6], dtype=np.float64)


import numpy as np
from scipy.optimize import root_scalar


def markets(x):
    # copy prices
    p = np.array([1.0, x[0]])
    w, r = x[1], x[2]

    # calculate total income
    Ybar = w * Lbar + r * Kbar
    PP = alpha * p[0]**(1 - nu) + (1 - alpha) * p[1]**(1 - nu)

    # get market equations
    market1 = 1 / p[0] - (beta[0] / w)**beta[0] * ((1 - beta[0]) / r)**(1 - beta[0])
    market2 = 1 / p[1] - (beta[1] / w)**beta[1] * ((1 - beta[1]) / r)**(1 - beta[1])
    market3 = beta[0] * alpha * Ybar / (w * p[0]**(nu - 1) * PP) + beta[1] * (1 - alpha) * Ybar / (w * p[1]**(nu - 1) * PP) - Lbar

    return np.array([market1, market2, market3])


# parameter values
Kbar = 10
Lbar = 20
alpha = 0.3
nu = 0.5
mu = 1 - 1 / nu
beta = np.array([0.3, 0.6])

# initial guess
x0 = np.array([0.5, 0.5, 0.5])

# find market equilibrium
sol = root_scalar(lambda x: markets(x), x0=x0, method='')

# check whether the solver converged
if not sol.converged:
    print("Error in solver!!!")
    exit()

# get equilibrium prices and factor prices
p = np.array([1.0, sol.root])
w, r = sol.root, sol.root

# calculate other economic variables
Ybar = w * Lbar + r * Kbar
PP = alpha * p[0]**(1 - nu) + (1 - alpha) * p[1]**(1 - nu)
Y = np.array([alpha * Ybar / (p[0]**nu * PP), (1 - alpha) * Ybar / (p[1]**nu * PP)])
L = beta * p * Y / w
K = (1 - beta) * p * Y / r
U = (alpha**(1 / nu) * Y[0]**mu + (1 - alpha)**(1 / nu) * Y[1]**mu)**(1 / mu)


# Output
print('Goods Market 1 :')
print('X1 = ', Y[0], 'Y1 = ', Y[0])
print('Goods Market 2 :')
print('X2 = ', Y[1], 'Y2 = ', Y[1])
print('Labour Market :')
print('L1 = ', L[0], 'L2 = ', L[1], 'L = ', Lbar)
print('Capital Market :')
print('K1 = ', K[0], 'K2 = ', K[1], 'K = ', Kbar)