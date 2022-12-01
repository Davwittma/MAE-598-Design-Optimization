import numpy as np
import torch as t
# outlining givens
def objective_f(x):
    f = x[0][0] ** 2 + (x[1][0] - 3) **2
    return f
def constraint_g1(x):
    g1 = x[1][0]**2 - 2*x[0][0] <= 0
    return g1
def constraint_g2(x, g, ):
    g2 = (x[1][0]-1) ** 2 + 5*x[0][0] -15 <=0
    return g2
#TODO: add derivatives of constraints and objective func

def linser(x, s, mu, w_p, k) #TODO: Finish Line search alg
    t = 0.3
    a=1

    if k ==0
        w==abs(mu)
    else:
        w = np.zeros((2,1))
        w[0] = max(abs(mu[0]), 0.5 * w_p[0] + abs(mu[0]))
        w[1] = max(abs(mu[1]), 0.5 * w_p[1 + abs(mu[1]))
        #TODO: add activating constraints

    def fa(x, w, a, s):
            #TODO: limits for constraints

#constructing SQP
def sqp(x, W)
    A0 = constraint_dg(x)
    b0 = constraint_g(x)
    mu0 = np.zeros((b0.shape[0], 1))
    mu = []
    active = []
    while True:
        if len(active) == 0:
            matrix = W
            s_mu = np.matmul(np.linalg.inv(matrix), -objective_df(x).T)
            s = s_mu[:2, :]
            mu = []

        if len(active) != 0:
            if len(active) == 1:
                A = A0[active[0], :].reshape(1, -1)
                b = b0[active[0], :]
            if len(active) == 2:
                A = copy.deepcopy(A0)
                b = copy.deepcopy(b0)
            matrix = np.vstack((np.hstack((W, A.T)),
                                np.hstack((A, np.zeros((A.shape[0], A.shape[0]))))))
            s_mu = np.matmul(np.linalg.inv(matrix), np.vstack((-objective_df(x).T, -b)))
            s = s_mu[:2, :]
            mu = s_mu[2:, :]
            if len(mu) == 1:
                mu0[0] = s_mu[2:3, :]
            if len(mu) == 2:
                mu0[0] = s_mu[2:3, :]
                mu0[1] = s_mu[3:, :]

        sqp_constraint = np.round((np.matmul(A0, s.reshape(-1, 1)) + b0))

        mu_check = 0

        if len(mu) == 0:
            mu_check = 1
        elif min(mu) > 0:
            mu_check = 1
        else:
            id_mu = np.argmin(np.array(mu))
            mu.remove(min(mu))
            active.pop(id_mu)

        if np.max(sqp_constraint) <= 0:
            if mu_check == 1:
                return s, mu0
        else:
            index = np.argmax(sqp_constraint)
            active.append(index)
            active = np.unique(np.array(active)).tolist()


    return
def BFGS(x, W, dx, s, mu):
    delt_L = (objectivedf(x) + np.matmul(mu.T, constraint_dg(x))) - (objectivedf(x-dx) + np.matmul(mu.T, constraintdg(x-dx)))




    return
