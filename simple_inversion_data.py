import numpy as np
import torch

v = np.array([1, 1], dtype="float32");
v.shape = (2,1)
def f(X):
    '''The function that we want to approximate using the network'''
    return np.linalg.inv(X) @ v

def generate_diag_elem():
    return np.random.uniform(0.5, 1.5)
def generate_angle():
    return np.random.uniform(0, 2*np.pi)

def generate_X():
    '''Generate invertible 2 by 2 matrix'''
    a = generate_angle()
    c, s = np.cos(a), np.sin(a)
    R = np.array([[c,-s], [s, c]])
    l1, l2 = generate_diag_elem(), generate_diag_elem()
    D = np.diag([l1, l2])
    return R @ D @ R.T

def generate_data(num_samples):
    '''Generate the learning data'''
    Xs = []
    ys = []
    for i in range(num_samples):
        X = generate_X()
        Xs.append(X)
        ys.append(f(X))
    return Xs, ys

def save_data(Xs, ys, filename):
    np.savez_compressed(filename, Xs=Xs, ys=ys)

def load_data(filename):
    loaded = np.load(filename)
    return torch.utils.data.TensorDataset(
        torch.Tensor(loaded["Xs"]), torch.Tensor(loaded["ys"]))
