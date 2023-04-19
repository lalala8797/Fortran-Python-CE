#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 22:58:49 2023

@author: khademul
"""

import numpy as np

def NormalDiscrete1(n, mu = 0.0, sigma = 1.0):
    """
    This function returns a discretized normal distribution N(mu,sigma) with
    n nodes and n corresponding weights.
    Input:  n : number of nodes
            mu: mean of required normal distribution, default is 0
            sigma: variance of required normal distribution, default is 1
    Output: x: value of each node
            prob: weight corresponding to each node
    """
    mu_c = mu            #mean of the distribution
    sigma_c = sigma**0.5 #standard deviation of the distribution

    if sigma_c < 0.0:
        raise ValueError("Sigma should have non-negative value")

    pim4 = 1.0/(np.pi**0.25) #square root of standard deviation parameter
    m = int((n+1)/2)     #Since normal distribution is symmetric, find out how many nodes one side included.

    #Initialize output variables
    x = np.zeros(n)      #initialized nodes' value
    prob = np.zeros(n)   #initialized nodes' weight

    z1 = 0.0             #middle value storing computed z

    for i in range(m):   #numerical approximation of normal distribution, ref: Fehr & Kindermann (2018) toolbox
        if i == 0:
            z = (2*n+1)**0.5-1.85575*(2*n+1)**(-1.0/6.0)
        elif i == 1:
            z = z - 1.14*(n**0.426)/z
        elif i == 2:
            z = 1.86*z + 0.86*x[0]
        elif i == 3:
            z= 1.91*z+0.91*x[1]
        else:
            z = 2.0*z+x[i-2]

        its = 0              #initial iteration number
        while its < 1000:
            its = its + 1
            p1 = pim4
            p2 = 0.0
            for j in range(1,n+1): # for j = 1, ..., n
                p3 = p2
                p2 = p1
                p1 = z*(2/j)**0.5*p2 - p3*((j-1)/j)**0.5

            pp = p2*(2.0*n)**0.5
            z1 = z
            z = z1-p1/pp
            if abs(z-z1)<1e-14:
                break

        if its>200:
            raise RuntimeError('Failed to converge')
        x[n-1-i] = z
        x[i] = -z
        prob[i]= 2.0/pp**2
        prob[n-1-i] = prob[i]

    prob = prob/np.pi**0.5 #normalization
    x = x*2.0**0.5*sigma_c + mu_c

    return x, prob

def NormalDiscrete2(n, mu = np.zeros(2), sigma = np.ones(2), rho = 0):
    '''
    This function creates n1*n2 pints and probabilities for a two-dimensional
    normal distribution

    Input:  n: [n1, n2], number of nodes
            mu: [mu1, mu2], mean of required normal distribution, default is 0
            sigma: [sig1, sig2], variance of required normal distribution, default is 1
            rho: correlation of two variables, default is 0
    return: x: [n1*n2, n1*n2] discrete points of normal distribution
            prob: n1*n2 array, probabiliteis of each realizeation
    '''

    mu_c = mu
    sig_c = sigma
    rho_c = rho

    sigma_c = np.zeros((2,2))

    sigma_c[0,0] = sig_c[0]
    sigma_c[1,1] = sig_c[1]
    sigma_c[0,1] = rho*(sig_c[0]*sig_c[1])**0.5
    sigma_c[1,0] = sigma_c[0,1]

    x = np.zeros((n[0]*n[1],2))
    prob = np.zeros(n[0]*n[1])

    if not np.allclose(sigma_c, sigma_c.T):
        raise ValueError('Variance-Covariance is not symmetric')

    x1, p1 = NormalDiscrete1(n=n[0])
    x2, p2 = NormalDiscrete1(n=n[1])

    m = 0

    for k in range(n[1]):
        for j in range(n[0]):
            prob[m] = p1[j]*p2[k]
            x[m,:] = [x1[j], x2[k]]
            m = m + 1
    if not (np.abs(sig_c) <= 1e-100 ).any() :
        l = np.linalg.cholesky(sigma_c)
    else:
        l = np.zeros((2,2))
        l[0,0] = sig_c[0]**0.5
        l[1,1] = sig_c[1]**0.5

    x = x@l.T
    x[:,0] += mu_c[0]
    x[:,1] += mu_c[1]

    return x, prob




def log_normal_discrete(n, mu = np.exp(0.5), sigma = np.exp(1.0)*(np.exp(1.0)-1.0)):
    """
    This function returns a discretized lognormal distribution logN(mu,sigma) with n nodes and n corresponding weights.
    Input:  n : number of nodes
            mu: mean of required lognormal distribution
            sigma: variance of required lognormal distribution
    Output: x: value of each node
            prob: weight corresponding to each node
    """
    mu_c = mu          #mean of distribution
    sigma_c = sigma    #standard deviation of distribution

    if sigma_c < 0.0:
        print('error: sigma has negative value')
    if mu_c <= 0.0:
        print('error: mu has zero or negative value')

    #Transfer from lognormal distribution to corresponding normal distribution
    sigma_c = np.log(1.0+sigma_c/mu_c**2) #mean of transfered normal distribution
    mu_c = np.log(mu_c)-0.5*sigma_c       #standard deviation of transfered normal distribution

    x = np.array(normal_discrete(n,mu_c,sigma_c))[0:1].reshape((n,)) #reshaping first column result to row
    x = np.exp(x) #transfer normal distributon discretized nodes to lognormal distribution values
    prob = np.array(normal_discrete(n,mu_c,sigma_c))[1:].reshape((n,)) #reshaping second column result to row
    return x, prob

## sourse: https://programtalk.com/vs2/python/3701/dolo/dolo/numeric/discretization/discretization.py/
def rouwenhorst(rho, sigma, N, weight = False):
    """
    Approximate an AR1 process by a finite markov chain using Rouwenhorst's method.

    :param rho: autocorrelation of the AR1 process
    :param sigma: conditional standard deviation of the AR1 process
    :param N: number of states
    :param weight: whether or not return unconditional stationary distribution weights
    :return [nodes, P]: equally spaced nodes and transition matrix (if weight = False)
    :return [nodes, P, w]: equally spaced nodes, transition matrix, unconditional stationary distribution weights(if weight = True)
    """

    from numpy import sqrt, linspace, array,zeros

    sigma = float(sigma)

    if N == 1:
        nodes = array([0.0])
        transitions = array([[1.0]])
        return [nodes, transitions]

    p = (rho+1)/2
    q = p
    nu = sqrt( (N-1)/(1-rho**2) )*sigma

    nodes = linspace( -nu, nu, N)
    sig_a = sigma
    n = 1
    #    mat0 = array( [[1]] )
    mat0 = array([[p,1-p],[1-q,q]])
    if N == 2:
        return [nodes,mat0]
    for n in range(3,N+1):
        mat = zeros( (n,n) )
        mat_A = mat.copy()
        mat_B = mat.copy()
        mat_C = mat.copy()
        mat_D = mat.copy()
        mat_A[:-1,:-1] = mat0
        mat_B[:-1,1:] = mat0
        mat_C[1:,:-1] = mat0
        mat_D[1:,1:] = mat0

        mat0 = p*mat_A + (1-p)*mat_B + (1-q)*mat_C + q*mat_D
        mat0[1:-1,:] = mat0[1:-1,:]/2
    P = mat0

    n = len(nodes)
    w = np.ones(n)*1/n
    for i in range(1,10001):
        w = P.T@w


    if weight:
        return [nodes, P, w]
    else:
        return [nodes, P]


def grow_grid(left, right, growth, n = 100):
    '''
    left: lower bound of the state space
    right: upper bounf of the state space
    growth: growth rate of the state space
    n: size of the state space, 100 by default
    '''
    assert left< right, 'left interval point should be less than right point'
    assert growth > 0, 'growth rate must be greater than zero'

    h = (right - left)/((1+growth)**(n-1) - 1)

    a = np.zeros(n)

    for i in range(n):
        a[i] = h*((1+growth)**i - 1) + left

    return a


def sort(a):
    '''
    Input: a: array like
    
    Output:
            asort: sorted array
            order: new ordering of the array
    '''
    asort = np.sort(a)
    order = []
    for i in range(len(asort)):
        index = np.where(a == asort[i])
        order = np.append(order, index).astype(int)
        
    for j in range(len(order)):
        if j <= len(order) - 1:
            indaux = np.where(order == order[j])[0]
            if len(indaux) != 1 :
                order = np.delete(order, indaux[1:])
    
    return asort, order