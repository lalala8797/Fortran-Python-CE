import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

# Global constants and parameters
JJ = 80  # Number of years the household lives
JR = 45  # Year the household retires
NA = 200  # Number of points on the asset grid
NP = 2  # Number of persistent shock process values
NS = 7  # Number of transitory shock process values
NW = 7  # Number of white noise shocks

gamma = 0.50
egam = 1.0 - 1.0 / gamma
beta = 0.98

sigma_theta = 0.242
sigma_eps = 0.022
rho = 0.985
sigma_white = 0.057

a_l = 0.0
a_u = 1000.0
a_grow = 0.05

is_initial = 4  # Define is_initial

# Initialize variables
dist_theta = np.zeros(NP)
theta = np.zeros(NP)
pi = np.zeros((NS, NS))
eta = np.zeros(NS)
dist_white = np.zeros(NW)
white = np.zeros(NW)
r = 0.04
w = 1.0
pen = np.zeros(JJ)
psi = np.zeros(JJ + 1)
eff = np.zeros(JJ)
a_bor = np.zeros((JJ, NP))
a = np.zeros(NA + 1)
aplus = np.zeros((JJ, NA + 1, NP, NS, NW))
c = np.zeros((JJ, NA + 1, NP, NS, NW))
phi = np.zeros((JJ, NA + 1, NP, NS, NW))
V = np.zeros((JJ, NA + 1, NP, NS, NW))
c_coh = np.zeros(JJ)
y_coh = np.zeros(JJ)
a_coh = np.zeros(JJ)
v_coh = np.zeros(JJ)
cv_c = np.zeros(JJ)
cv_y = np.zeros(JJ)
RHS = np.zeros((JJ, NA + 1, NP, NS))
EV = np.zeros((JJ, NA + 1, NP, NS))
cons_com = 0.0

ij_com, ia_com, ip_com, is_com, iw_com = 0, 0, 0, 0, 0

def foc(x_in):
    global cons_com

    a_plus = x_in
    wage = w * eff[ij_com] * theta[ip_com] * eta[is_com] * white[iw_com]
    available = (1.0 + r) * (a[ia_com] + a_bor[ij_com, ip_com]) + wage + pen[ij_com]
    cons_com = available - (a_plus + a_bor[ij_com + 1, ip_com])

    il, ir, varphi = linint_Grow(a_plus, a_l, a_u, a_grow, NA)
    tomorrow = varphi * RHS[ij_com + 1, il, ip_com, is_com] + (1.0 - varphi) * RHS[ij_com + 1, ir, ip_com, is_com]

    return cons_com - tomorrow

def margu(cons):
    return cons**(-1.0 / gamma)

def valuefunc(a_plus, cons, ij, ip, is_):
    c_help = np.maximum(cons, 1e-10)
    il, ir, varphi = linint_Grow(a_plus, a_l, a_u, a_grow, NA)

    if ij < JJ-1:
        tomorrow_value = np.maximum(varphi * EV[ij + 1, il, ip, is_] + (1.0 - varphi) * EV[ij + 1, ir, ip, is_], 1e-10)
        future_value = tomorrow_value**egam / egam
    else:
        future_value = 0.0

    return c_help**egam / egam + beta * psi[ij + 1] * future_value

def initialize():
    global r, w, psi, eff, pen, dist_theta, theta, eta, pi, white, dist_white, a, a_bor

    psi[:] = [1.0, 0.99923, 0.99914, 0.99914, 0.99912, 0.99906, 0.99908, 0.99906, 0.99907, 0.99901,
              0.99899, 0.99896, 0.99893, 0.99890, 0.99887, 0.99886, 0.99878, 0.99871, 0.99862, 0.99853,
              0.99841, 0.99835, 0.99819, 0.99801, 0.99785, 0.99757, 0.99735, 0.99701, 0.99676, 0.99650,
              0.99614, 0.99581, 0.99555, 0.99503, 0.99471, 0.99435, 0.99393, 0.99343, 0.99294, 0.99237,
              0.99190, 0.99137, 0.99085, 0.99000, 0.98871, 0.98871, 0.98721, 0.98612, 0.98462, 0.98376,
              0.98226, 0.98062, 0.97908, 0.97682, 0.97514, 0.97250, 0.96925, 0.96710, 0.96330, 0.95965,
              0.95619, 0.95115, 0.94677, 0.93987, 0.93445, 0.92717, 0.91872, 0.91006, 0.90036, 0.88744,
              0.87539, 0.85936, 0.84996, 0.82889, 0.81469, 0.79705, 0.78081, 0.76174, 0.74195, 0.72155, 0.0]

    eff[:JR-1] = [1.0000, 1.0719, 1.1438, 1.2158, 1.2842, 1.3527, 1.4212, 1.4897, 1.5582, 1.6267, 
                  1.6952, 1.7217, 1.7438, 1.7748, 1.8014, 1.8279, 1.8545, 1.8810, 1.9075, 1.9341, 
                  1.9606, 1.9623, 1.9640, 1.9658, 1.9675, 1.9692, 1.9709, 1.9726, 1.9743, 1.9760, 
                  1.9777, 1.9700, 1.9623, 1.9546, 1.9469, 1.9392, 1.9315, 1.9238, 1.9161, 1.9084, 
                  1.9007, 1.8354, 1.7701, 1.7048]
    eff[JR:JJ] = 0.0

    pen[JR:JJ] = 0.5 * w * sum(eff) / (JR - 1)

    dist_theta[:] = 1.0 / NP
    theta[0] = np.exp(-np.sqrt(sigma_theta))
    theta[1] = np.exp(np.sqrt(sigma_theta))

    eta, pi = discretize_AR(rho, 0.0, sigma_eps)
    eta = np.exp(eta)

    white, dist_white = normal_discrete(0.0, sigma_white)
    white = np.exp(white)

    a[:] = np.geomspace(a_l + 1e-10, a_u, NA + 1)
    a_bor = np.zeros((JJ, NP))

    for ij in range(1, JJ):
        for ip in range(NP):
            abor_temp = 0.0
            for ijj in range(JR, ij, -1):
                abor_temp = abor_temp / (1.0 + r) + eff[ijj] * theta[ip] * eta[0] * white[0] + pen[ijj]
            abor_temp = min(-abor_temp / (1.0 + r) + 1e-4, 0.0)
            a_bor[ij, ip] = max(a_bor[ij, ip], abor_temp)

def solve_household():
    global aplus, c, V

    for ia in range(NA + 1):
        aplus[JJ-1, ia, :, :, :] = 0.0
        c[JJ-1, ia, :, :, :] = (1.0 + r) * (a[ia] + a_bor[JJ-1, 0]) + pen[JJ-1]
        V[JJ-1, ia, :, :, :] = valuefunc(0.0, c[JJ-1, ia, 0, 0, 0], JJ-1, 0, 0)

    interpolate(JJ-1)

    for ij in range(JJ-2, -1, -1):
        ip_max = 1 if ij >= JR else NP
        is_max = 1 if ij >= JR else NS
        iw_max = 1 if ij >= JR else NW

        for ia in range(NA + 1):
            if ij >= JR and ia == 0 and pen[ij] <= 1e-10:
                aplus[ij, ia, :, :, :] = 0.0
                c[ij, ia, :, :, :] = 0.0
                V[ij, ia, :, :, :] = valuefunc(0.0, 0.0, ij, 0, 0)
                continue

            for ip in range(ip_max):
                for is_ in range(is_max):
                    for iw in range(iw_max):
                        x_in = aplus[ij + 1, ia, ip, is_, iw]

                        global ij_com, ia_com, ip_com, is_com, iw_com
                        ij_com, ia_com, ip_com, is_com, iw_com = ij, ia, ip, is_, iw

                        res = minimize_scalar(foc, bounds=(a_l, a_u), method='bounded')
                        if not res.success:
                            print(f"ERROR IN ROOTFINDING : {ij} {ia} {ip} {is_} {iw}")

                        x_in = max(res.x, 0.0)
                        if x_in < 0.0:
                            x_in = 0.0
                            wage = w * eff[ij] * theta[ip] * eta[is_] * white[iw]
                            available = (1.0 + r) * (a[ia] + a_bor[ij, ip]) + wage + pen[ij]
                            cons_com = available - a_bor[ij + 1, ip]

                        aplus[ij, ia, ip, is_, iw] = x_in
                        c[ij, ia, ip, is_, iw] = cons_com
                        V[ij, ia, ip, is_, iw] = valuefunc(x_in, cons_com, ij, ip, is_)

            if ij >= JR:
                aplus[ij, ia, :, :, :] = aplus[ij, ia, 0, 0, 0]
                c[ij, ia, :, :, :] = c[ij, ia, 0, 0, 0]
                V[ij, ia, :, :, :] = V[ij, ia, 0, 0, 0]

        interpolate(ij)

        print(f"Age: {ij} DONE!")

def interpolate(ij):
    global RHS, EV

    for ia in range(NA + 1):
        for ip in range(NP):
            for is_ in range(NS):
                RHS[ij, ia, ip, is_] = 0.0
                EV[ij, ia, ip, is_] = 0.0

                for is_p in range(NS):
                    for iw in range(NW):
                        chelp = max(c[ij, ia, ip, is_p, iw], 1e-10)
                        RHS[ij, ia, ip, is_] += pi[is_, is_p] * dist_white[iw] * margu(chelp)
                        EV[ij, ia, ip, is_] += pi[is_, is_p] * dist_white[iw] * V[ij, ia, ip, is_p, iw]

                RHS[ij, ia, ip, is_] = (beta * psi[ij] * (1.0 + r) * RHS[ij, ia, ip, is_])**(-gamma)
                EV[ij, ia, ip, is_] = (egam * EV[ij, ia, ip, is_])**(1.0 / egam)

def get_distribution():
    global phi

    phi[:] = 0.0

    for ip in range(NP):
        il, ir, varphi = linint_Grow(-a_bor[0, ip], a_l, a_u, a_grow, NA)
        for iw in range(NW):
            phi[0, il, ip, is_initial, iw] = varphi * dist_theta[ip] * dist_white[iw]
            phi[0, ir, ip, is_initial, iw] = (1.0 - varphi) * dist_theta[ip] * dist_white[iw]

    for ij in range(1, JJ):
        for ia in range(NA + 1):
            for ip in range(NP):
                for is_ in range(NS):
                    for iw in range(NW):
                        il, ir, varphi = linint_Grow(aplus[ij - 1, ia, ip, is_, iw], a_l, a_u, a_grow, NA)
                        il = min(il, NA)
                        ir = min(ir, NA)
                        varphi = min(varphi, 1.0)

                        for is_p in range(NS):
                            for iw_p in range(NW):
                                phi[ij, il, ip, is_p, iw_p] += pi[is_, is_p] * dist_white[iw_p] * varphi * phi[ij - 1, ia, ip, is_, iw]
                                phi[ij, ir, ip, is_p, iw_p] += pi[is_, is_p] * dist_white[iw_p] * (1.0 - varphi) * phi[ij - 1, ia, ip, is_, iw]

def aggregation():
    global c_coh, y_coh, a_coh, v_coh, cv_c, cv_y

    c_coh[:] = 0.0
    y_coh[:] = 0.0
    a_coh[:] = 0.0
    v_coh[:] = 0.0

    for ij in range(JJ):
        for ia in range(NA + 1):
            for ip in range(NP):
                for is_ in range(NS):
                    for iw in range(NW):
                        c_coh[ij] += c[ij, ia, ip, is_, iw] * phi[ij, ia, ip, is_, iw]
                        y_coh[ij] += eff[ij] * theta[ip] * eta[is_] * white[iw] * phi[ij, ia, ip, is_, iw]
                        a_coh[ij] += (a[ia] + a_bor[ij, ip]) * phi[ij, ia, ip, is_, iw]
                        v_coh[ij] += V[ij, ia, ip, is_, iw] * phi[ij, ia, ip, is_, iw]

    cv_c[:] = 0.0
    cv_y[:] = 0.0

    for ij in range(JJ):
        for ia in range(NA + 1):
            for ip in range(NP):
                for is_ in range(NS):
                    for iw in range(NW):
                        cv_c[ij] += c[ij, ia, ip, is_, iw]**2 * phi[ij, ia, ip, is_, iw]
                        cv_y[ij] += (eff[ij] * theta[ip] * eta[is_] * white[iw])**2 * phi[ij, ia, ip, is_, iw]

    cv_c = np.sqrt(cv_c - c_coh**2) / c_coh
    cv_y = np.sqrt(cv_y - y_coh**2) / np.maximum(y_coh, 1e-10)

def output():
    global c_coh, y_coh, a_coh, pen, cv_c, cv_y

    iamax = np.zeros(JJ, dtype=int)
    check_grid(iamax)

    ages = np.arange(20, 20 + JJ)

    with open('output.out', 'w') as f:
        f.write(f"{'IJ':>3} {'CONS':>10} {'EARNINGS':>10} {'INCOME':>10} {'PENS':>10} {'ASSETS':>10} {'A_BOR(1)':>10} {'A_BOR(2)':>10} {'CV(C)':>10} {'CV(L)':>10} {'VALUE':>10} {'IAMAX':>10}\n")
        for ij in range(JJ):
            f.write(f"{ij+1:3} {c_coh[ij]:10.3f} {y_coh[ij]:10.3f} {(y_coh[ij] + r * a_coh[ij]):10.3f} {pen[ij]:10.3f} {a_coh[ij]:10.3f} {a_bor[ij, 0]:10.3f} {a_bor[ij, 1]:10.3f} {cv_c[ij]:10.3f} {cv_y[ij]:10.3f} {v_coh[ij]:10.3f} {iamax[ij]:10.3f}\n")

    plt.plot(ages, c_coh, label='Consumption')
    plt.plot(ages, y_coh + pen, label='Earnings')
    plt.xlabel('Age j')
    plt.ylabel('Mean')
    plt.legend()
    plt.show()

    plt.plot(ages, a_coh)
    plt.xlabel('Age j')
    plt.ylabel('Assets')
    plt.show()

    plt.plot(ages, cv_c, label='Consumption')
    plt.plot(ages, cv_y, label='Earnings')
    plt.xlabel('Age j')
    plt.ylabel('Coefficient of Variation')
    plt.legend()
    plt.show()

    frac_bor = np.zeros(JJ)
    for ij in range(JJ - 1):
        for ia in range(NA + 1):
            for ip in range(NP):
                for is_ in range(NS):
                    for iw in range(NW):
                        if aplus[ij, ia, ip, is_, iw] < 1e-6:
                            frac_bor[ij] += phi[ij, ia, ip, is_, iw]

    frac_bor[JJ - 1] = 1.0

    plt.plot(ages[:-1], frac_bor[:-1])
    plt.xlabel('Age j')
    plt.ylabel('Frac. Borrowing Constrained Households')
    plt.show()

def linint_Grow(x, left, right, growth, n):
    xinv = grid_Inv_Grow(np.clip(x, left, right), left, right, growth, n)
    il = np.clip(np.floor(xinv).astype(int), 0, n - 1)
    ir = il + 1
    h = (right - left) / ((1 + growth)**n - 1)
    xl = h * ((1 + growth)**il - 1) + left
    xr = h * ((1 + growth)**ir - 1) + left
    phi = (xr - x) / (xr - xl)
    return il, ir, phi

def grid_Inv_Grow(x, left, right, growth, n):
    h = (right - left) / ((1 + growth)**n - 1)
    return np.log((x - left) / h + 1) / np.log(1 + growth)

def discretize_AR(rho, mu, sigma):
    eta = np.linspace(-3, 3, NS) * sigma / np.sqrt(1 - rho ** 2)
    pi = np.exp(-0.5 * ((eta - mu) / sigma) ** 2)
    pi /= np.sum(pi)
    return eta, pi

def normal_discrete(mu, sigma):
    white = np.linspace(-3, 3, NW) * sigma + mu
    dist_white = np.exp(-0.5 * ((white - mu) / sigma) ** 2)
    dist_white /= np.sum(dist_white)
    return white, dist_white

def check_grid(iamax):
    global phi

    for ij in range(JJ):
        iamax[ij] = 0
        for ia in range(NA + 1):
            for ip in range(NP):
                for is_ in range(NS):
                    for iw in range(NW):
                        if phi[ij, ia, ip, is_, iw] > 1e-8:
                            iamax[ij] = ia

initialize()
print("Initialization complete.")
solve_household()
get_distribution()
aggregation()
output()
