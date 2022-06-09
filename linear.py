
import numpy as np
import math as m


def GridInvGrow(x, left, right, growth, n):
    '''
    Calculate inverse of gridpoints of a growing grid
    x: point that shall be calculated
    left: left interval point
    right: right interval point
    n: number of grid points (0,1,2,...,n)
    '''
    assert left < right, 'Left interval point must less than right interval point'
    assert growth > 0, 'Growth rate must be greater than zero'
    n = n - 1
    h = (right -left)/((1+growth)**n - 1)
    grid = np.log((x-left)/h + 1)/np.log(1+growth)
    return grid


def LinintGrow(x, left, right, growth, n):
    '''
    Calculates linear interpolant on a growing grid
    x: point that shall be calculated
    left: left interval point
    right: right interval point
    n: number of grid points (0,1,2,...,n)

    Return:
    il: left interpolation point
    ir: right interpolation point
    ϕ: interpolation fraction
    '''
    assert left < right, 'Left interval point must less than right interval point'
    assert growth > 0, 'Growth rate must be greater than zero'

    n = n - 1
    xinv = GridInvGrow(min(max(x,left),right), left, right, growth, n+1)
    il = min(max(m.floor(xinv),0),n-1)
    ir = il + 1
    h = (right-left) / ((1 + growth)**n - 1)
    xl = h*((1+growth)**il - 1) + left
    xr = h*((1+growth)**ir - 1) + left
    ϕ = (xr - x)/(xr-xl)
    return il, ir, ϕ


def GridInvEqui(x, left, right, n):
    '''
    Calculates inverse of gridpoints of an equidistant gridpoints
    Input:
        x: point that shall be calculated
        left: left interval point
        right: right interval point
        n: number of grid points
    '''
    assert left < right, 'Left interval point must less than right interval point'
    h = (right - left)/(n-1)
    grid = (x-left)/h
    return grid


def LinintEqui(x, left, right, n):
    '''
    Calclate linear interpolation on an equidistant grid.
    Input:
        x: point that shall be Calculate
        left: left interval point
        right: right interval point
        n: number of grid points
    Out:
        il: left interpolation point
        ir: right interpolation point
        phi: interpolation fraction
    '''

    n = n - 1

    xinv = GridInvEqui(min(max(x, left), right), left, right, n+1)

    il = min(max(m.floor(xinv), 0), n - 1)
    ir = il + 1
    h = (right - left)/n
    xl = h*il + left
    xr = h*ir + left

    phi = (xr - x)/(xr - xl)
    return il, ir, phi
