import numpy as np
import matplotlib.pyplot as plt
from discretize_tools import rouwenhorst, grow_grid
from scipy.optimize import root
from linear import linint
import time
import math as m

class LifeCycleFemale:

    '''
    This class specifies the life cycle model with female participation choice
    '''

    def __init__(self,
                γ = 0.5, β = 0.98, r = 0.04, wm = 1.0, wf = 0.75, J = 80, JR = 45,
                NP = 2, NS = 5, NA = 81, NH = 61, ν = 0.12, σθ = 0.242, σϵ = 0.022,
                ρ = 0.985, al = 0, au = 450, agrow = 0.05,
                ξ = [0.05312, -0.00188], δh = 0.074, hgrow = 0.03, hl =0,
                children = np.concatenate(([30,32], np.zeros(8))), verbose = True,
                ψ = np.array([1.00000, 0.99923, 0.99914, 0.99914, 0.99912, \
                0.99906, 0.99908, 0.99906, 0.99907, 0.99901, \
                0.99899, 0.99896, 0.99893, 0.99890, 0.99887, \
                0.99886, 0.99878, 0.99871, 0.99862, 0.99853, \
                0.99841, 0.99835, 0.99819, 0.99801, 0.99785, \
                0.99757, 0.99735, 0.99701, 0.99676, 0.99650, \
                0.99614, 0.99581, 0.99555, 0.99503, 0.99471, \
                0.99435, 0.99393, 0.99343, 0.99294, 0.99237, \
                0.99190, 0.99137, 0.99085, 0.99000, 0.98871, \
                0.98871, 0.98721, 0.98612, 0.98462, 0.98376, \
                0.98226, 0.98062, 0.97908, 0.97682, 0.97514, \
                0.97250, 0.96925, 0.96710, 0.96330, 0.95965, \
                0.95619, 0.95115, 0.94677, 0.93987, 0.93445, \
                0.92717, 0.91872, 0.91006, 0.90036, 0.88744, \
                0.87539, 0.85936, 0.84996, 0.82889, 0.81469, \
                0.79705, 0.78081, 0.76174, 0.74195, 0.72155, \
                0.00000]),
                eff = np.array([1.0000, 1.0719, 1.1438, 1.2158, 1.2842, 1.3527, \
                1.4212, 1.4897, 1.5582, 1.6267, 1.6952, 1.7217, \
                1.7438, 1.7748, 1.8014, 1.8279, 1.8545, 1.8810, \
                1.9075, 1.9341, 1.9606, 1.9623, 1.9640, 1.9658, \
                1.9675, 1.9692, 1.9709, 1.9726, 1.9743, 1.9760, \
                1.9777, 1.9700, 1.9623, 1.9546, 1.9469, 1.9392, \
                1.9315, 1.9238, 1.9161, 1.9084, 1.9007, 1.8354, \
                1.7701, 1.7048])):

        self.γ, self.β, self.ν, self.r, self.wm, self.wf, self.ψ = γ, β, ν, r, wm, wf, ψ
        self.J, self.JR, self.NA, self.NP, self.NS, self.NH = J, JR, NA, NP, NS, NH
        self.egam, self.children, self.verbose = 1-1/γ, children, verbose
        self.al, self.au, self.agrow, self.hl, self.hgrow, self.ξ, self.δh  = al, au, agrow, hl, hgrow, ξ, δh
        self.ρ, self.σθ, self.σϵ = ρ, σθ, σϵ

        # asset grid
        self.a = grow_grid(al, au, agrow, NA)

        # surving probability and labor efficiency and pensions
        self.eff = np.concatenate((eff, np.zeros(J - JR +1)))
        self.pen = np.concatenate((np.zeros(JR-1), np.ones(J-JR+1)*(0.8*eff.sum()/(JR - 1))))

        # initialize the human capital grid by defining human capital maximum
        self.hmax = np.zeros(self.J)
        for j in range(1, JR-1):
            self.hmax[j] = self.hmax[j-1] + ξ[0] + ξ[1]*j
        self.hmax[JR-1:] = self.hmax[JR-2]
        self.hu = np.amax(self.hmax)
        self.h = grow_grid(hl, self.hu, hgrow, NH)

        # children number and children price
        self.nc = np.zeros(J) # number of children
        self.pc = np.zeros(J) # price of caring chilren
        price = wf * np.exp(self.hmax[9])
        # the price of one unit of childcare is equal to the wage of a 30-year-old woman who has worked full-time
        # between the ages 20 to 30 (in our index system, the index is '9')

        for n in range(10):
            if children[n] > 20 and children[n] < 50:

                ## calculate age child is born
                j = int(children[n] - 21) ## we need to subtract 21 to get the index of the age of chid born

                # set number of children
                self.nc[j:j+18] += 1

                # set cost of child care
                self.pc[j:j+3] += price
                self.pc[j+3:j+6] += 0.8*price
                self.pc[j+6:j+12] += 0.6*price
                self.pc[j+12:j+18] += 0.4*price

        # persistent shock grid
        self.θ = np.exp(np.array([-1,1])*self.σθ**0.5)
        self.θprob = np.ones(self.NP)*(1/self.NP)

        # stochastic shock grid
        self.η, self.π = rouwenhorst(ρ, σϵ**0.5, NS)
        self.η = np.exp(self.η)
        self.middle = int((self.NS + 1)/2) - 1


        self.aplus, self.c, self.l, self.V = self.policy()
        self.Φ = self.distribution()
        self.c_coh, self.ym_coh, self.yf_coh, self.l_coh, self.a_coh, self.h_coh, self.v_coh, self.cv_c, self.cv_y = self.aggregation()



    def policy(self):
        if self.verbose:
            print('Here comes the policy function part')
        global RHS, EV

        aplus = np.zeros((self.J, self.NA, self.NH, self.NP, self.NS, self.NS))
        c = aplus.copy()
        l = aplus.copy()
        V = aplus.copy()
        utemp = np.zeros(2)
        ctemp = np.zeros(2)
        aptemp = np.zeros(2)

        RHS = np.zeros((self.NA, self.NH, self.NP, self.NS, self.NS, 2))
        EV = np.zeros((self.NA, self.NH, self.NP, self.NS, self.NS, 2))

        margu = lambda x, y: x**(-1/self.γ)/((2 + self.nc[y])**0.5)**self.egam

        def interpolate(ij):
            global RHS, EV

            for i in range(self.NA):
                if ij > self.JR - 1:
                    ihmax = 1
                    pmax = 1
                    smax = 1
                    lmax = 1
                else:
                    ihmax = self.NH
                    pmax = self.NP
                    smax = self.NS
                    lmax = 2

                for ih in range(ihmax):
                    for p in range(pmax):
                        for sm in range(smax):
                            for sf in range(smax):
                                for la in range(lmax):

                                    RHS[i,ih,p,sm,sf,la] = 0
                                    EV[i,ih,p,sm,sf,la] = 0

                                    ## interpolate human capital for tomorrow
                                    hplus = max(self.h[ih] + (self.ξ[0] + self.ξ[1]*ij)*la - self.δh*(1-la), self.hl)

                                    ihl, ihr, φ2 = linint(hplus, self.hl, self.hu, self.hgrow, self.NH)

                                    # iterate over all potential future states

                                    for ism in range(self.NS):
                                        caux = φ2*c[ij,i,ihl,p,ism,:] + (1-φ2)*c[ij,i,ihr,p,ism,:]
                                        caux = np.maximum(caux, 1e-10)
                                        RHS[i,ih,p,sm,sf,la] += self.π[sm,ism]*(self.π[sf,:]@margu(caux,ij))

                                        # expected value function
                                        Vhelp = np.maximum(φ2 * (self.egam*V[ij,i,ihl,p,ism,:]) ** (1/self.egam) + \
                                         (1-φ2) * (self.egam*V[ij,i,ihr,p,ism,:]) ** (1/self.egam), 1e-10)**self.egam/self.egam

                                        EV[i,ih,p,sm,sf,la] += self.π[sm,ism]*(self.π[sf,:]@Vhelp)

                                    RHS[i,ih,p,sm,sf,la] = ((1+self.r)*self.β*self.ψ[ij]*RHS[i,ih,p,sm,sf,la])**(-self.γ)
                                    EV[i,ih,p,sm,sf,la] = (self.egam*EV[i,ih,p,sm,sf,la])**(1/self.egam)




        def foc(ap):
            wagem = self.wm*self.eff[j]*self.θ[p]*self.η[sm]
            wagef = self.wf*la*(np.exp(self.h[ih])*self.θ[p]*self.η[sf] - self.pc[j])

            consum = (1+self.r)*self.a[i] + self.pen[j] + wagef + wagem - ap
            ap = max(ap, self.al)

            il, ir, φ1 = linint(ap, self.al, self.au, self.agrow, self.NA)
            tomorrow = φ1*RHS[il, ih, p, sm, sf, la] + (1-φ1)* RHS[ir, ih, p, sm, sf, la]
            foc = consum/((2 + self.nc[j])**0.5)**(1-self.γ) - tomorrow

            return foc


        def VF(ap, c, lab, j, ih, p, sm, sf):

            cons = max(c, 1e-10)

            il, ir, φ0 = linint(ap, self.al, self.au, self.agrow, self.NA)

            vf = 0

            if j < self.J-1:
                vf = max(φ0*EV[il,ih,p,sm,sf,lab] + (1-φ0)*EV[ir,ih,p,sm,sf,lab], 1e-10)**self.egam/self.egam

            vf = (cons/((2 + self.nc[j])**0.5))**self.egam/self.egam - self.ν*lab + self.β*self.ψ[j+1]*vf

            return vf



        for i in range(self.NA):
            aplus[-1,i,:,:,:,:] = 0
            c[-1,i,:,:,:,:] = (1 + self.r) * self.a[i] + self.pen[-1]
            l[-1,i,:,:,:,:] = 0
            V[-1,i,:,:,:,:] = VF(0, c[-1,i,0,0,0,0], 0, self.J-1, 0, 0, 0, 0)

        interpolate(self.J-1)

        for j in range(self.J-2, -1, -1):
            # check about how many states to iterate

            if j >= self.JR - 1:
                ihmax = 1
                pmax = 1
                smax = 1
                lmax = 1
                utemp[1] = -1e+100
            else:
                ihmax = self.NH
                pmax = self.NP
                smax = self.NS
                lmax = 2

            for i in range(self.NA):

                if j >=self.JR-1 and i == 0 and self.pen[j] <= 1e-10:
                    aplus[j,i,:,:,:,:] = 0 ## tomorrow's asset is obviously zero
                    c[j,i,:,:,:,:] = 0 ## also, consumption is obviously zero as there is no income at all
                    l[j,i,:,:,:,:] = 0 ## same as above
                    V[j,i,:,:,:,:] = VF(0, 0, 0, j, 0, 0, 0, 0)
                    continue

                for ih in range(ihmax):
                    # check whether h is greater than hmax[j]
                    # if the current h[ih] is greater than hmax[j], than we don't need to calculate the decision
                    # rule for this one, just copy & paste (dosen't matter what value we set it)
                    if self.h[ih] > self.hmax[j]:
                        aplus[j,i,ih,:,:,:] = aplus[j,i,ih-1,:,:,:]
                        c[j,i,ih,:,:,:] = c[j,i,ih-1,:,:,:]
                        l[j,i,ih,:,:,:] = l[j,i,ih-1,:,:,:]
                        V[j,i,ih,:,:,:] = V[j,i,ih-1,:,:,:]
                        continue

                    for p in range(pmax):
                        for sm in range(smax):
                            for sf in range(smax):
                                # determine solution for both working decisions
                                for la in range(lmax): # l = 0, 1 or l = 0
                                    ap = root(foc, x0 = aplus[j+1,i,ih,p,sm,sf]).x[0]
                                    if ap < 0:
                                        ap = 0
                                    wagem = self.wm*self.eff[j]*self.θ[p]*self.η[sm]
                                    wagef = self.wf*la*(np.exp(self.h[ih])*self.θ[p]*self.η[sf] - self.pc[j])
                                    aptemp[la] = ap
                                    ctemp[la] = (1+self.r)*self.a[i] + self.pen[j] + wagef + wagem - aptemp[la]
                                    utemp[la] = VF(aptemp[la], ctemp[la], la, j, ih, p, sm, sf)

                                # choose labor force status that gives more utility

                                if utemp[1] >= utemp[0]:
                                    aplus[j,i,ih,p,sm,sf] = aptemp[1]
                                    c[j,i,ih,p,sm,sf] = ctemp[1]
                                    l[j,i,ih,p,sm,sf] = 1
                                    V[j,i,ih,p,sm,sf] = utemp[1]
                                else:
                                    aplus[j,i,ih,p,sm,sf] = aptemp[0]
                                    c[j,i,ih,p,sm,sf] = ctemp[0]
                                    l[j,i,ih,p,sm,sf] = 0
                                    V[j,i,ih,p,sm,sf] = utemp[0]

                if j >= self.JR - 1:
                    aplus[j,i,:,:,:,:] = aplus[j,i,0,0,0,0]
                    c[j,i,:,:,:,:] = c[j,i,0,0,0,0]
                    l[j,i,:,:,:,:] = l[j,i,0,0,0,0]
                    V[j,i,:,:,:,:] = V[j,i,0,0,0,0]

            interpolate(j)
            if self.verbose:
                print('P: Period = {}'.format(j))



        return aplus, c, l, V


    def distribution(self):
        if self.verbose:
            print('Here comes the distribution part')

        Φ = np.zeros((self.J,self.NA,self.NH,self.NP,self.NS,self.NS))

        for p in range(self.NP):
            Φ[0,0,0,p,2,2] = 0.5

        for j in range(1,self.J):

            for i in range(self.NA):
                for ih in range(self.NH):
                    for p in range(self.NP):
                        for sm in range(self.NS):
                            for sf in range(self.NS):


                                il, ir, φa = linint(self.aplus[j-1,i,ih,p,sm,sf], self.al, self.au, self.agrow, self.NA)

                                il = min(il,self.NA)
                                ir = min(ir,self.NA)
                                φa = min(φa, 1)

                                labor = self.l[j-1,i,ih,p,sm,sf]
                                htoday = self.h[ih] + (self.ξ[0] + self.ξ[1]*j)*labor - self.δh*(1-labor)
                                htoday = max(htoday, self.hl)

                                ihl,ihr, φh = linint(htoday, self.hl, self.hu, self.hgrow, self.NH)

                                ihl = min(ihl,self.NH)
                                ihr = min(ihr,self.NH)
                                φh = min(φh, 1)

                                if j >= self.JR - 1:
                                    ihl = ih
                                    ihr = ih

                                for ism in range(self.NS):
                                    Φ[j,il,ihl,p,ism,:] += self.π[sm,ism]*self.π[sf,:]*φa*φh*Φ[j-1,i,ih,p,sm,sf]
                                    Φ[j,ir,ihl,p,ism,:] += self.π[sm,ism]*self.π[sf,:]*(1-φa)*φh*Φ[j-1,i,ih,p,sm,sf]
                                    Φ[j,il,ihr,p,ism,:] += self.π[sm,ism]*self.π[sf,:]*φa*(1-φh)*Φ[j-1,i,ih,p,sm,sf]
                                    Φ[j,ir,ihr,p,ism,:] += self.π[sm,ism]*self.π[sf,:]*(1-φa)*(1-φh)*Φ[j-1,i,ih,p,sm,sf]
            if self.verbose:
                print('D: Period = {}'.format(j))
        return Φ


    def aggregation(self):
        if self.verbose:
            print('Here comes the aggregation part')
        c_coh = np.zeros(self.J)
        ym_coh = np.zeros(self.J)
        yf_coh = np.zeros(self.J)
        l_coh = np.zeros(self.J)
        a_coh = np.zeros(self.J)
        h_coh = np.zeros(self.J)
        v_coh = np.zeros(self.J)

        for j in range(self.J):
            for i in range(self.NA):
                for ih in range(self.NH):
                    for p in range(self.NP):
                        for sm in range(self.NS):
                            for sf in range(self.NS):
                                wagem = self.wm*self.eff[j]*self.θ[p]*self.η[sm]
                                wagef = self.wf*np.exp(self.h[ih])*self.θ[p]*self.η[sf]*self.l[j,i,ih,p,sm,sf]
                                c_coh[j] += self.c[j,i,ih,p,sm,sf]*self.Φ[j,i,ih,p,sm,sf]
                                ym_coh[j] += wagem*self.Φ[j,i,ih,p,sm,sf]
                                yf_coh[j] += wagef*self.Φ[j,i,ih,p,sm,sf]
                                l_coh[j] += self.l[j,i,ih,p,sm,sf]*self.Φ[j,i,ih,p,sm,sf]
                                a_coh[j] += self.a[i]*self.Φ[j,i,ih,p,sm,sf]
                                h_coh[j] += np.exp(self.h[ih])*self.Φ[j,i,ih,p,sm,sf]
                                v_coh[j] += self.V[j,i,ih,p,sm,sf]*self.Φ[j,i,ih,p,sm,sf]
            if self.verbose:
                print('Acoh: Period = {}'.format(j))

        cv_c = np.zeros(self.J)
        cv_y = np.zeros(self.J)
        for j in range(self.J):
            for i in range(self.NA):
                for ih in range(self.NH):
                    for p in range(self.NP):
                        for sm in range(self.NS):
                            for sf in range(self.NS):
                                wagem = self.wm*self.eff[j]*self.θ[p]*self.η[sm]
                                wagef = self.wf*np.exp(self.h[ih])*self.θ[p]*self.η[sf]*self.l[j,i,ih,p,sm,sf]
                                cv_c[j] += self.c[j,i,ih,p,sm,sf]**2*self.Φ[j,i,ih,p,sm,sf]
                                cv_y[j] += (wagem+wagef)**2*self.Φ[j,i,ih,p,sm,sf]
            if self.verbose:
                print('Acv: Period = {}'.format(j))

        cv_c = (cv_c - c_coh**2)**0.5/c_coh
        cv_y = (cv_y - (ym_coh+yf_coh)**2)**0.5/np.maximum(ym_coh+yf_coh, 1e-10)

        return c_coh, ym_coh, yf_coh, l_coh, a_coh, h_coh, v_coh, cv_c, cv_y

if __name__ == '__main__':
    model = LifeCycleFemale()
