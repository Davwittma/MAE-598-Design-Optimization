import numpy as np
from scipy.optimize import minimize


def objective_fcn(x):
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]
    x5 = x[4]
    return (x1 - x2) ** 2 + (x2 + x3 - 2) ** 2 + (x4 - 1) ** 2 + (x5 - 1) ** 2


def eqconstraint1(x):
    x1 = x[0]
    x2 = x[1]
    return x1 + 3 * x2


def eqconstraint2(x):
    x3 = x[2]
    x4 = x[3]
    x5 = x[4]
    return x3 + x4 - 2 * x5


def eqconstraint3(x):
    x2 = x[1]
    x5 = x[4]
    return x2 - x5


bounds_x1 = (-10, 10)
bounds_x2 = (-10, 10)
bounds_x3 = (-10, 10)
bounds_x4 = (-10, 10)
bounds_x5 = (-10, 10)
bounds = [bounds_x1, bounds_x2, bounds_x3, bounds_x4, bounds_x5]
constraint1 = {'type': 'eq', 'fun': eqconstraint1}
constraint2 = {'type': 'eq', 'fun': eqconstraint2}
constraint3 = {'type': 'eq', 'fun': eqconstraint3}
constraint = [constraint1, constraint2, constraint3]
x0 = [1, 1, 1, 1, 1]
res = minimize(objective_fcn, x0, method='SLSQP', bounds=bounds, constraints=constraint)
print(res)
breakpoint()
