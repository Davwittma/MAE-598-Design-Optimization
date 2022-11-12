# P4
import numpy as np
import torch as t

# Formulating components of equation for df/dd
# 1 x1 is decision and x2/3 are state variable

def obj(x1, x2, x3):
    return x1 ** 2 + x2 ** 2 + x3 ** 2


def constraint(x1, x2, x3):
    h1 = x1 ** 2 / 4 + x2 ** 2 / 5 + x3 ** 2 / 25 - 1
    h2 = x1 + x2 - x3
    return h1, h2


def dfdd(x1, x2, x3):
    df_ds, df_dd = df_dx(x1, x2, x3)
    dh_ds_inv, dh_dd, _ = dh_dx(x1, x2, x3)
    return df_dd - np.matmul(df_ds.T, np.matmul(dh_ds_inv, dh_dd))


def df_dx(x1, x2, x3):
    df_ds = np.array([[2 * x2], [2 * x3]])
    df_dd = np.array([2 * x1], dtype=float)
    return df_ds, df_dd


def dh_dx(x1, x2, x3):
    dh_ds = np.vstack((np.hstack(((2 / 5) * x2, (2 / 25) * x3)), np.array([[1, -1]], dtype=float))) #TODO add Pytorch functionality
    dh_ds_inv = np.linalg.inv(dh_ds)
    dh_dd = np.vstack(((1 / 2) * x1, 1.))
    return dh_ds_inv, dh_dd, dh_ds

# Using Solver
def solveh(x1, x2, x3):
    error = 1e-3
    # check if the constraint is satisfied
    h1, h2 = constraint(x1, x2, x3)
    h = np.vstack((h1, h2))
    h_norm = np.linalg.norm(h)  # inverse again
    while h_norm >= error:
        dh_inv, _, dh = dh_dx(x1, x2, x3)
        Lambda = 1
        ds = np.matmul(dh_inv, h)
        x2 = x2 - ds[0]
        x3 = x3 - ds[1]
        h1, h2 = constraint(x1, x2, x3)
        h = np.vstack((h1, h2))
        h_norm = np.linalg.norm(h)
    return x1, x2, x3, h_norm


def line_search(x1, x2, x3):
    a = 1.  # initialize step size
    df = dfdd(x1, x2, x3)
    phi = lambda a, x1, x2, x3, df: obj(x1, x2, x3) - a * 0.3 * np.matmul(df, df.T)  # define phi as a search criterion

    def f_a(x1, x2, x3, a):
        df = dfdd(x1, x2, x3)
        dh_ds_inv, dh_dd, _ = dh_dx(x1, x2, x3)

        x1 = x1 - a * df.flatten()
        ds = np.matmul(np.matmul(dh_ds_inv, dh_dd), df.T).flatten()
        x2 = x2 + a * ds[0]
        x3 = x3 + a * ds[1]
        return obj(x1, x2, x3)

    while phi(a, x1, x2, x3, df) < f_a(x1, x2, x3, a):
        a = 0.5 * a
        df = dfdd(x1, x2, x3)
    return a
