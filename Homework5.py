import numpy as np
import torch as t

def objective_f(x):
    f = x[0][0] ** 2 + (x[1][0] - 3) **2
    return f
def constraint_g1(x):
    g1 = x[1][0]**2 - 2*x[0][0] <= 0
    return g1
def constraint_g2(x, g, ):
    g2 = (x[1][0]-1) ** 2 + 5*x[0][0] -15 <=0
    return g2


def sqp(x, )




    return


def BFGS()



    return
