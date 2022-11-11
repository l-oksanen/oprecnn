import numpy as np
import torch

v = np.array([1, 1], dtype="float32");
v.shape = (2,1)
def f(X):
    '''The function that we want to approximate using the network'''
    return np.linalg.inv(X) @ v

def generate_diag_elem():
    sign = 1-2*np.random.randint(0,1) # gives 1 or -1, sign of a eigenvalue
    return sign*np.random.uniform(0.5, 1.5)
def generate_angle():
    return np.random.uniform(0, 2*np.pi)

def generate_X():
    '''Generate invertible 2 by 2 matrix'''
    a = generate_angle()
    c, s = np.cos(a), np.sin(a)
    R = np.array([[c,-s], [s, c]])
    l1, l2 = generate_diag_elem(), generate_diag_elem() # eigenvalues
    D = np.diag([l1, l2])
    return R @ D @ R.T

def generate_data(num_samples):
    # luodaan lista matriiseja ja niiden käänteismatriiseja
    '''Generate the learning data'''
    Xs = []
    ys = []
    for i in range(num_samples):
        X = generate_X() # luodaan kääntyvä matriisi
        Xs.append(X) # lisätään tämä luotu matriisi X matriisijonoon
        ys.append(f(X)) #lisätään käännetty matriisi X käännettyjen jonoon
    return Xs, ys # palautetaan lista/jono matriiseja ja niiden käänteismatriiseja

def save_data(Xs, ys, filename):
    np.savez_compressed(filename, Xs=Xs, ys=ys)

def load_data(filename):
    loaded = np.load(filename)
    return torch.utils.data.TensorDataset(
        torch.Tensor(loaded["Xs"]), torch.Tensor(loaded["ys"]))
