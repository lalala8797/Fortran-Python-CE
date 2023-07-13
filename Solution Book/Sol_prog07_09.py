import numpy as np

# Model parameters
TT = 36
JJ = 6
JR = 5
gamma = 0.5
egam = 1.0 - 1.0 / gamma
beta = 0.9
alpha = 0.3
delta = 0.0
delh = 0.0
xi = 5.0
upsi1 = 0.2
upsi2 = 0.7
epsi = 0.0
mu = 1.0
tol = 1e-6
damp = 0.4
itermax = 100

# Model variables
w = np.zeros(TT+1)
r = np.zeros(TT+1)
wn = np.zeros(TT+1)
winc = np.zeros((JJ+1, TT+1))
Rn = np.zeros(TT+1)
p = np.zeros(TT+1)
tauw = np.zeros(TT+1)
taur = np.zeros(TT+1)
tauc = np.zeros(TT+1)
taup = np.zeros(TT+1)
pen = np.zeros((JJ+1, TT+1))
by = np.zeros(TT+1)
kappa = np.zeros(TT+1)
nnp = np.zeros(TT+1)
gy = 0.0
tauk = np.zeros(TT+1)
KK = np.zeros(TT+1)
LL = np.zeros(TT+1)
YY = np.zeros(TT+1)
AA = np.zeros(TT+1)
CC = np.zeros(TT+1)
II = np.zeros(TT+1)
BB = np.zeros(TT+1)
GG = np.zeros(TT+1)
BF = np.zeros(TT+1)
TB = np.zeros(TT+1)
Tpen = np.zeros(TT+1)
TXR = np.zeros(TT+1)
m = np.zeros((JJ+1, TT+1))
ma = np.zeros((JJ+1, TT+1))
a = np.zeros((JJ+1, TT+1))
c = np.zeros((JJ+1, TT+1))
util = np.zeros((JJ+1, TT+1))
taus = np.zeros(TT+1)
e = np.zeros(TT+1)
HH = np.zeros(TT+1)
ne = np.zeros(TT+1)
XX = np.zeros(TT+1)
pin = np.zeros(TT+1)
tax = np.zeros(TT+1, dtype=int)
eps = np.zeros(TT+1, dtype=int)
smopec = True
it = 0


def initialize():
    global np, gy, smopec, by, pin, kappa, eps, tauk, taus, tax, tauc, tauw, taur, taup, a, e, wn, winc, ne, YY, HH, BF, TB, TXR, MA, XX, m

    nnp[0] = 0.0
    nnp[1:] = 0.0
    gy = 0.0
    smopec = True

    by[0:] = 0.0
    pin[0] = 0.005
    pin[1:] = 0.02
    kappa[0] = 0.0
    kappa[1:] = 0.0
    eps[0] = 0
    eps[1:] = 0
    tauk[0] = 0.0
    tauk[1:] = 0.0
    taus[0] = 0.0
    taus[1:] = 0.0
    tax[0] = 1
    tax[1:] = 1

    tauc[0:] = 0.0
    tauw[0:] = 0.0
    taur[0:] = 0.0
    taup[0:] = 0.0
    pen[0:, 0:] = 0.0

    a[0:, 0:] = 0.0
    e[0:] = 0.0
    wn[0:] = 0.0
    winc[0:, 0:] = 0.0
    ne[0:] = 0.0
    YY[0:] = 0.0
    HH[0:] = 1.0
    BF[0:] = 0.0
    TB[0:] = 0.0
    TXR[0:] = 0.0
    ma[0:, 0:] = 0.0
    XX[0:] = 0.0

    for it in range(TT+1):
        m[1, it] = 1.0
        itm = year(it, 2, 1)
        for j in range(2, JJ+1):
            m[j, it] = m[j-1, itm] / (1.0 + nnp[itm])


def get_SteadyState():
    iter = 0

    KK[0] = 1.0
    LL[0] = 1.0

    while iter < itermax:
        factor_prices(0)
        decisions(0)
        quantities(0)
        government(0)

        if abs(YY[0] - CC[0] - II[0] - GG[0] - XX[0]) / YY[0] < tol:
            break

        iter += 1

    if iter < itermax:
        print(f"Iteration: {iter} Diff: {abs(YY[0] - CC[0] - II[0] - GG[0] - XX[0]) / YY[0]}")
    else:
        print("Convergence not achieved!")

    output_summary(0)


def factor_prices(it):
    itm = year(it, 2, 1)
    w[it] = (1.0 - tauw[it]) * (1.0 - tauc[it]) * (1.0 - tauk[it]) * (YY[it] - BB[it]) / (LL[it] * HH[it])
    wn[it] = (1.0 - taus[it]) * e[it] * wn[it]

    if smopec and it > 0:
        r[it] = (1.0 - tauk[it]) * (YY[it] - w[it] * LL[it]) / BB[it]
    else:
        r[it] = r[itm]

    for j in range(1, JJ+1):
        winc[j, it] = (1.0 - tauw[it]) * (1.0 - tauc[it]) * (1.0 - tauk[it]) * (YY[it] - w[it] * LL[it]) / m[j, it]


def decisions(it):
    getedu(it)
    get_path(1, it)

    if smopec and it > 0:
        KK[it] = LL[it] * ((r[it] * (1.0 - eps[it] * tauk[it]) / (1.0 - tauk[it]) + delta) / alpha)**(1.0 / (alpha - 1.0))
        BF[it] = AA[it] - KK[it] - BB[it]

    for j in range(2, JJ+1):
        get_path(j, it)

    if smopec:
        II[it] = (1.0 + nnp[it]) * (1.0 + ne[it]) * KK[year(it, 1, 2)] - (1.0 - delta) * KK[it]
    else:
        KK[it] = damp * (AA[it] - BB[it]) + (1.0 - damp) * KK[it]
        II[it] = (1.0 + nnp[it]) * (1.0 + ne[it]) * KK[year(it, 1, 2)] - (1.0 - delta) * KK[it]


def getedu(it):
    pvw = 0.0
    PRn = 1.0

    for j in range(2, JR):
        itp = year(it, 1, j)
        PRn *= Rn[itp]
        pvw += wn[itp] * (1.0 - delh)**(j-2) / PRn

    e[it] = (upsi1 * xi * XX[it]**upsi2 * pvw / (wn[it] * (1.0 - taus[it])))**(1.0 / (1.0 - upsi1))
    if e[it] > 0.99:
        e[it] = 0.99

    itp = year(it, 1, 2)
    ne[itp] = mu * (1.0 + xi * e[it]**upsi1 * XX[it]**upsi2) - 1.0


def get_path(j, it):
    PRn = 1.0

    for jp in range(j+1, JJ+1):
        itp = year(it, j, jp)
        PRn *= Rn[itp]
        itm = year(it, j, jp-1)
        c[jp, itp] = (beta**(jp-j) * PRn * p[it] / p[itp])**gamma * c[j, it]
        a[jp, itp] = winc[jp-1, itm] + pen[jp-1, itm] + Rn[itm] * a[jp-1, itm] - p[itm] * c[jp-1, itm]


def quantities(it):
    itm = year(it, 2, 1)

    if it == 0:
        GG[it] = gy * YY[it]
    else:
        GG[it] = GG[0]

    ma[1, it] = 1.0

    for j in range(2, JJ+1):
        ma[j, it] = ma[j-1, itm] / (1.0 + ne[it]) / (1.0 + nnp[it])

    CC[it] = 0.0
    AA[it] = 0.0
    LL[it] = 1.0 - e[it]
    HH[it] = 0.0

    for j in range(1, JJ+1):
        itp = year(it, j, 2)
        CC[it] += c[j, it] * ma[j, it]
        AA[it] += a[j, it] * ma[j, it]
        if j > 1 and j < JR:
            LL[it] += (1.0 - delh)**(j-2) * (1.0 + ne[itp]) / mu * ma[j, it]
        if j < JR:
            HH[it] += m[j, it]

    HH[it] = LL[it] / (HH[it] - e[it])
    YY[it] = KK[it]**alpha * LL[it]**(1.0 - alpha) * HH[it]**epsi
    BB[it] = by[itm] * YY[it]

    itp = year(it, 1, 2)

    if smopec and it > 0:
        KK[it] = LL[it] * ((r[it] * (1.0 - eps[it] * tauk[it]) / (1.0 - tauk[it]) + delta) / alpha)**(1.0 / (alpha - 1.0))
        BF[it] = AA[it] - KK[it] - BB[it]
        TB[it] = (1.0 + nnp[itp]) * (1.0 + ne[itp]) * BF[itp] - (1.0 + r[it]) * BF[it]
    else:
        KK[it] = damp * (AA[it] - BB[it]) + (1.0 - damp) * KK[it]
        II[it] = (1.0 + nnp[itp]) * (1.0 + ne[itp]) * KK[itp] - (1.0 - delta) * KK[it]


def government(it):
    itp = year(it, 1, 2)
    XX[it] = pin[it] * YY[it]

    taxrev = np.zeros(5)
    taxrev[0] = tauc[it] * CC[it]
    taxrev[1] = tauw[it] * w[it] * LL[it]
    taxrev[2] = taur[it] * r[it] * AA[it]
    taxrev[3] = tauk[it] * (YY[it] - w[it] * LL[it] - (delta + eps[it] * r[it]) * KK[it])
    taxrev[4] = taus[it] * e[it] * wn[it]
    PEXP = taxrev[4] + (1.0 + r[it]) * BB[it] + GG[it] + XX[it]

    if tax[it] == 1:
        tauc[it] = (PEXP - (taxrev[1] + taxrev[2] + taxrev[3] + (1.0 + nnp[itp]) * (1.0 + ne[itp]) * BB[itp])) / CC[it]
    elif tax[it] == 2:
        tauw[it] = (PEXP - (taxrev[0] + taxrev[3] + (1.0 + nnp[itp]) * (1.0 + ne[itp]) * BB[itp])) / (w[it] * LL[it] + r[it] * AA[it])
        taur[it] = tauw[it]
    elif tax[it] == 3:
        tauw[it] = (PEXP - (taxrev[0] + taxrev[2] + taxrev[3] + (1.0 + nnp[itp]) * (1.0 + ne[itp]) * BB[itp])) / (w[it] * LL[it])
    else:
        taur[it] = (PEXP - (taxrev[0] + taxrev[1] + taxrev[3] + (1.0 + nnp[itp]) * (1.0 + ne[itp]) * BB[itp])) / (r[it] * AA[it])

    TXR[it] = np.sum(taxrev[0:4]) - taxrev[4]

    pen[JR:JJ, it] = kappa[it] * winc[JR-1, it]
    Tpen[it] = np.sum(pen[JR:JJ, it] * ma[JR:JJ, it])
    taup[it] = Tpen[it] / w[it] / LL[it]


def output(it, file):
    diff = YY[it] - CC[it] - GG[it] - II[it] - TB[it] - XX[it]

    print(f"Equilibrium: Year {it}", file=file)
    print("Goods Market", file=file)
    print("      Y      C      G      I     TB    INF      DIFF", file=file)
    print(f"{YY[it]:.2f} {CC[it]:.2f} {GG[it]:.2f} {II[it]:.2f} {TB[it]:.2f} {XX[it]:.2f} {diff:.4f}", file=file)
    print("      Y      C      G      I     TB    INF", file=file)
    print(f"{YY[it]/YY[it]:.2f} {CC[it]/YY[it]:.2f} {GG[it]/YY[it]:.2f} {II[it]/YY[it]:.2f} {TB[it]/YY[it]:.2f} {XX[it]/YY[it]:.2f}", file=file)
    print("Capital Market", file=file)
    print("      A      K     BB     BF       r", file=file)
    print(f"{AA[it]:.2f} {KK[it]:.2f} {BB[it]:.2f} {BF[it]:.2f} {r[it]:.2f}", file=file)
    print("      A      K     BB     BF", file=file)
    print(f"{AA[it]/YY[it]:.2f} {KK[it]/YY[it]:.2f} {BB[it]/YY[it]:.2f} {BF[it]/YY[it]:.2f}", file=file)
    print("Labor Market", file=file)
    print("     LL      e     ne      w   util", file=file)
    print(f"{LL[it]:.2f} {e[it]:.3f} {ne[it]:.2f} {w[it]:.2f} {util[1, it]:.2f}", file=file)
    print("GOVERNMENT", file=file)
    print("   tauc   tauw   taur   taup   tauk   taus    TXR     DD     rB", file=file)
    #print(f"{tauc[it]:.2f} {tauw[it]:.2f} {taur[it]:.2f} {taup[it]:.2f} {tauk[it]:.2f} {taus[it]:.2f} {TXR[it]/YY[it]:.2f} {(1+np[it1])*BB[it1]-BB[it]/YY[it]:.2f} {r[it]*BB[it]/YY[it]:.2f}", file=file)

    if file > 1:
        print("Age    cons       wn      pen    asset    Diff", file=file)
        for j in range(1, JJ+1):
            it1 = year(it, 1, 2)
            if j < JJ:
                diff = Rn[it] * a[j, it] + winc[j, it] + pen[j, it] - a[j+1, it1] - p[it] * c[j, it]
                print(f"{j:3d} {c[j, it]:.2f} {winc[j, it]:.2f} {pen[j, it]:.2f} {a[j, it]:.2f} {diff:.2f}", file=file)
            else:
                diff = Rn[it] * a[j, it] + pen[j, it] - p[it] * c[j, it]
                print(f"{j:3d} {c[j, it]:.2f}  0.00 {pen[j, it]:.2f} {a[j, it]:.2f} {diff:.2f}", file=file)
        print("", file=file)


def output_summary(file):
    print("      pin      C      e     ne      L      K      r      w      Y      B     BA   tauc   tauw   taur   taup   tauk   taus    HEV   y-d", file=file)

    for it in range(TT+1):
        diff = YY[it] - CC[it] - GG[it] - II[it] - XX[it]
        print(f"{it:3d} {pin[it]:7.4f} {CC[it]:7.2f} {e[it]:7.3f} {ne[it]:7.2f} {LL[it]:7.2f} {KK[it]:7.2f} {r[it]:7.2f} {w[it]:7.2f} {YY[it]:7.2f} {BB[it]:7.2f} {(util[1, it]/util[1, 0])**(1.0/egam)-1.0:7.2f} {tauc[it]:7.2f} {tauw[it]:7.2f} {taur[it]:7.2f} {taup[it]:7.2f} {tauk[it]:7.2f} {taus[it]:7.2f} {(util[2, it]/util[2, 0])**(1.0/egam)-1.0:7.4f} {diff:7.4f}", file=file)


def year(it, j, jp):
    itm = it - JR + jp
    if itm < 1:
        itm = it - j + 1
    return itm


def util_summary():
    utility = np.zeros((JJ+1, TT+1))
    summary = np.zeros((4, TT+1))

    for j in range(1, JJ+1):
        for it in range(1, TT+1):
            utility[j, it] = (1.0 - tauc[it]) * c[j, it] - kappa[it] * winc[j-1, it]
        summary[1, it] += utility[j, it] * ma[j, it] * (1.0 + nnp[it])
        summary[2, it] += (1.0 - taus[it]) * utility[j, it] * wn[it] * ma[j, it]
        summary[3, it] += utility[j, it] * a[j, it] * ma[j, it]

    for it in range(1, TT+1):
        summary[0, it] = summary[1, it] + summary[2, it] + summary[3, it]

    return summary


def main():
    initialize()
    get_SteadyState()


if __name__ == '__main__':
    main()
