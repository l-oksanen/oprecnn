import numpy as np
import torch
from scipy.interpolate import KroghInterpolator

import wave1d

def generate_c_vals(num_cells):
    '''Generate the constants values of c'''
    return np.random.uniform(0.5, 1.5, num_cells)



def bump(x, a=0, b=0.4, radius=0.2, deg=3):
    a0 = a
    b0 = b
    a1 = a0 + radius
    b1 = b0 - radius

    assert a1 <= b1, 'a + radius must be less than b - radius'

    pts1 = np.concatenate((a0*np.ones(deg), a1*np.ones(deg)))
    vals1 = np.zeros(2*deg)
    vals1[deg] = 1
    up = KroghInterpolator(pts1, vals1)

    pts2 = np.concatenate((b1*np.ones(deg), b0*np.ones(deg)))
    vals2 = np.zeros(2*deg)
    vals2[0] = 1
    down = KroghInterpolator(pts2, vals2)

    return np.piecewise(x, [
        np.logical_and(x > a0, x < a1),
        np.logical_and(x >= a1, x <= b1),
        np.logical_and(x > b1, x < b0)
        ], [up, 1, down, 0])

def compute_Lambda_h(c):
    def a(x):
        return c(x)**2
    Lambda_h = []
    def save_step(un, t, x, n):
        Lambda_h.append(un[0])
    dt = 0.02
    T = 2.5
    _, ts, xs = wave1d.solver(dt, T, a=a, h0=bump, cmax=1.5, user_action=save_step)
    return np.array(Lambda_h), ts

def time_translate(x):
    n = len(x)
    X = np.zeros((n,n))
    for i in range(n):
        if i == 0:
            X[ :,0] = x
        else:
            X[i:,i] = x[:-i]
    return X



def cell_condlist(x, cells):
    n = np.size(cells) - 1
    condlist = []
    for i in range(n):
        condlist.append(np.logical_and(x >= cells[i], x < cells[i+1]))
    return condlist

def pw_const(cells, vals):
    '''Create function from piecewise values'''
    def fun(x):
        return np.piecewise(x, cell_condlist(x, cells), vals)   
    return fun    

def integral_pw_const_node_vals(cells, vals):
    '''Compute the values of the integral of a piecewise constant function on cell boundaries'''
    n = np.size(cells) - 1
    out = np.zeros(n + 1)
    for i in range(1, n + 1):
        dx = cells[i] - cells[i-1]
        out[i] = out[i-1] + dx * vals[i-1]
    return out

def integral_pw_const(cells, vals):
    '''Create integral function from piecewise values'''
    n = np.size(cells) - 1
    node_vals = integral_pw_const_node_vals(cells, vals)
    def piece(i):
        return lambda x: node_vals[i] + (x - cells[i]) * vals[i]
    pieces = [piece(i) for i in range(n)]
    def fun(x):
        return np.piecewise(x, cell_condlist(x, cells), pieces)  
    return fun

def inverse_pw_lin(cells, node_vals):
    '''Create inverse function of a piecewise linear function given by node values'''
    n = np.size(cells) - 1
    def piece(i):
        dy = node_vals[i+1] - node_vals[i]
        dx = cells[i + 1] - cells[i]
        return lambda y: (y - node_vals[i]) * dx / dy + cells[i]
    pieces = [piece(i) for i in range(n)]
    def fun(y):
        return np.piecewise(y, cell_condlist(y, node_vals), pieces) 
    return fun

def c(cells, c_vals):
    return pw_const(cells, c_vals)

def tau(cells, c_vals):
    return integral_pw_const(cells, 1/c_vals)

def chi(cells, c_vals):
    tau_node_vals = integral_pw_const_node_vals(cells, 1/c_vals)
    return inverse_pw_lin(cells, tau_node_vals)

def V(cells, c_vals):
    Vtilde = integral_pw_const(cells, 1/c_vals**2)
    chi_fixed = chi(cells, c_vals)
    def fun(x):
        return Vtilde(chi_fixed(x))
    return fun



def generate_data(num_samples):
    '''Generate the learning data'''

    num_cells = 5
    cells = np.linspace(0, 1, num_cells+1)
    xs = []
    ys = []
    for i in range(num_samples):
        c_vals = generate_c_vals(num_cells)
        Lambda_h, _ = compute_Lambda_h(c(cells, c_vals))
        xs.append(Lambda_h)
        ys.append(V(cells, c_vals)(0.5))
    return np.array(xs), np.array(ys)

def save_data(xs, ys, filename):
    np.savez_compressed(filename, xs=xs, ys=ys)

def load_data(filename):
    loaded = np.load(filename)
    # Turn Lambda_h (a vector) into Lambda (a matrix) by using translations in time
    xs = loaded["xs"]
    Xs = []    
    for x in xs:
        Xs.append(time_translate(x))
    Xs = np.array(Xs)
    return torch.utils.data.TensorDataset(
        torch.Tensor(Xs), torch.Tensor(loaded["ys"]))
