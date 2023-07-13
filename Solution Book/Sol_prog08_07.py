import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar

# Model parameters
gamma = 0.5
egam = 1 - 1 / gamma
beta = 0.975
r = 0.02
w = 1
a0 = 0
a_borrow = 5

# Numerical parameters
sig = 1e-6
itermax = 2000

# Time path of consumption and resource
TT = 200
c_t = np.zeros(TT+1)
a_t = np.zeros(TT+1)

# Value and policy function
NA = 1000
a = np.zeros(NA+1)
c = np.zeros(NA+1)
V = np.zeros(NA+1)
V_new = np.zeros(NA+1)
coeff_V = np.zeros(NA+3)
coeff_c = np.zeros(NA+3)
con_lev = 0.0
x_in = 0.0
fret = 0.0
ia_com = 0

# Function to be minimized
def utility(x_in):
    global a, r, w, beta, egam, ia_com

    # Calculate consumption
    cons = (1 + r) * a[ia_com] + w - x_in

    # Calculate future utility
    vplus = max(spline_eval(x_in, coeff_V, a_l, a_u), 1e-10)**egam / egam

    # Calculate utility function
    if cons < 1e-10:
        utility = -1e-10**egam / egam * (1 + abs(cons))
    else:
        utility = -(cons**egam / egam + beta * vplus)

    return utility

# Grid definition
def grid_Cons_Equi(a, a_l, a_u):
    a[0] = a_l
    a[NA] = a_u
    for ia in range(1, NA):
        a[ia] = a_l + (ia - 1) / (NA - 1) * (a_u - a_l)

# Spline interpolation
def spline_interp(x, coeff):
    f = interp1d(a, x, kind='cubic')
    coeff[:] = f.c

# Spline evaluation
def spline_eval(x_in, coeff, a_l, a_u):
    f = interp1d(a, coeff, kind='cubic')
    return f(x_in)

# Minimization routine
def fminsearch(x_in, fret, a_l, a_u, utility):
    res = minimize_scalar(utility, bounds=(a_l, min((1 + r) * a[ia_com] + w, a_u)))
    x_in = res.x
    fret = res.fun

# Output plotting routine
def output():
    global a, c, NA, a_l, a_u, TT, c_t, a_t

    # Interpolate policy function
    spline_interp(c, coeff_c)

    # Calculate time path of consumption with borrowing constraint
    a_t[0] = a0
    c_t[0] = spline_eval(a_t[0], coeff_c, a_l, a_u)
    for it in range(1, TT+1):
        a_t[it] = (1 + r) * a_t[it-1] + w - c_t[it-1]
        c_t[it] = spline_eval(a_t[it], coeff_c, a_l, a_u)

    # Plot consumption with borrowing constraint
    plot(range(TT+1), c_t, legend='with borr. constraint')

    # Calculate time path of consumption without borrowing constraint (analytically)
    a_t[0] = a0
    c_t[0] = (a_t[0] + w / r) * (1 + r - (beta * (1 + r))**gamma)
    for it in range(1, TT+1):
        a_t[it] = (1 + r) * a_t[it-1] + w - c_t[it-1]
        c_t[it] = (a_t[it] + w / r) * (1 + r - (beta * (1 + r))**gamma)

    # Plot consumption without borrowing constraint
    plot(range(TT+1), c_t, legend='without borr. constraint')
    execplot(xlabel='Time t', ylabel='Consumption c_t')

    # Plot consumption
    plot(a, c, legend='with borr. constraint')
    plot(a, (a + w / r) * (1 + r - (beta * (1 + r))**gamma), legend='without borr. constraint')
    execplot(xlabel='Level of resources a', ylabel='Policy Function c(a)')

    # Plot value function
    plot(a[10:NA], V[10:NA], legend='with borr. constraint')
    plot(a[10:NA], ((1 + r)**(1 - gamma) - beta**gamma)**(-1 / gamma) *
         (a[10:NA] + w / r)**(1 - 1 / gamma) / (1 - 1 / gamma), legend='without borr. constraint')
    execplot(xlabel='Level of resources a', ylabel='Value Function V(a)')

    # Quit program
    quit()

# Plotting function
def plot(x, y, color='blue', linewidth=2.0, linestyle='-', marker='o', markersize=5, label=None):
    plt.plot(x, y, color=color, linewidth=linewidth, linestyle=linestyle, marker=marker, markersize=markersize, label=label)

# Execute plot
def execplot(xlabel='', ylabel=''):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()

# Tic and toc functions for timing
import time

tic_time = 0.0

def tic():
    global tic_time
    tic_time = time.time()

def toc():
    elapsed_time = time.time() - tic_time
    print("Elapsed time: {:.2f} seconds".format(elapsed_time))

# Main program
def main():
    global NA, itermax, sig, r, w, a, c, V, V_new, coeff_V, coeff_c, con_lev, a_l, a_u, x_in, fret, ia_com

    # Start timer
    tic()

    # Get lower and upper bound of grid
    a_l = max(-w / r, -a_borrow)
    a_u = w / r

    # Initialize a, c, and value function
    grid_Cons_Equi(a, a_l, a_u)
    c[:] = (r * a[:] + w) / 2
    V[:] = 0
    coeff_V[:] = 0

    # Iterate until value function converges
    for iter in range(1, itermax + 1):
        # Calculate optimal decision for every grid point
        for ia in range(0, NA + 1):
            # Initialize starting value and communicate resource level
            x_in = a[ia] * (1 + r) + w - c[ia]
            ia_com = ia

            # Call fminsearch
            fminsearch(x_in, fret, a_l, min((1 + r) * a[ia] + w, a_u), utility)

            # Get optimal consumption and value function
            c[ia] = a[ia] * (1 + r) + w - x_in
            V_new[ia] = -fret

        # Interpolate coefficients
        #spline_interp((egam * V_new)**(1 / egam), coeff_V)
        spline_interp(V_new[1:NA+1], coeff_V)


        # Get convergence level
        con_lev = np.max(np.abs(V_new[:] - V[:]) / np.maximum(np.abs(V[:]), 1e-10))
        print("{:5d}  {:20.7f}".format(iter, con_lev))

        # Check for convergence
        if con_lev < sig:
            output()

        V[:] = V_new[:]

    print("No Convergence")

if __name__ == '__main__':
    main()
