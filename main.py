from numba import jit
import numpy as np
from scipy.stats import chi2,norm
import matplotlib.pyplot as plt
import pandas as pd
import math




@jit
def create_cov(n=1000):

    cov = np.zeros(shape=(n,n))

    for i in range(n):
        for j in range(n):
            if(j<=i):
                cov[i,j] = 1/(n)**0.5
    return cov


def create_brown_motion_i(n=1000,i = 1,m = 1):

    dzeta = np.random.normal(size=n)
    len_dz = np.sqrt(np.sum(dzeta**2))

    X = dzeta/len_dz
    U = np.random.uniform()
    U = (i-1)/m+1/m*U
    D = np.sqrt(chi2.ppf(U,n))
    Z = D*X
    A = create_cov(n=n)
    B = np.matmul(A,Z)
    return B


def create_geom_motion(B = None,n=1000,r = 0.05 ,sd = 0.25,S_0 = 100,T =1):
    if B is None:
        B = create_brown_motion_i(n=n)
    n = len(B)
    t = np.arange(1/n,T+1/(2*n),1/n)
    S = S_0*np.exp((r-sd**2/2)*t+sd*B)
    return S




def Black_shoes_form(r = 0.05,sd = 0.25,S_0 = 100,K = 100):
    d1 = 1/sd*(r+sd**2/2+np.log(S_0/K))
    d2 = d1-sd
    return S_0*norm.cdf(d1)-K*np.exp(-r)*norm.cdf(d2)

def generate_and_calculate_I(n = 100,r = 0.05,sd = 0.25,S_0 = 100,i =1,m =1,K = 100,for_Control_variates = False):
    B = create_brown_motion_i(n=n,i = i,m = m)
    S = create_geom_motion(B = B,n=n,r = r ,sd = sd,S_0 = S_0,T = 1)
    A = np.mean(S)
    I = np.exp(-r) * max(0, A - K)
    if for_Control_variates:
        return [I,B[-1]]
    return I


###CMC est
def calculate_CMC_est(R = 100,n = 100,r = 0.05,sd = 0.25,S_0 = 100,K = 100):
    v = np.array([generate_and_calculate_I(n =n,r =r,sd = sd,S_0 = S_0,K = K) for _ in range(R)])
    return np.mean(v)

def calculate_CMC_est_vec(N = 10,R = 100,n = 1,r = 0.05,sd = 0.25,S_0 = 100,K = 100):
    return np.array([calculate_CMC_est(R = R,n =n,r =r,sd = sd,S_0 = S_0,K = K) for _ in range(N)])



###Strat est


def calculate_Strat_est(ps,Ri,R = 100,n = 100,m=1,r = 0.05,sd = 0.25,S_0 = 100,K = 100):

    CMCs = np.zeros(m)
    for i in range(m):
        v = np.array([generate_and_calculate_I(n=n, r=r, sd=sd, S_0=S_0,i = i+1,m = m, K=K) for _ in range(Ri[i])])
        CMCs[i] = np.mean(v)
    est = np.sum(CMCs*ps)
    return est

def calculate_Strat_prop_est_vec(N = 10,m = 10,R = 100,n = 1,r = 0.05,sd = 0.25,S_0 = 100,K = 100):
    ps = np.zeros(m) + 1 / m
    Ri = np.ceil(ps * R).astype(int)
    return np.array([calculate_Strat_est(ps = ps,Ri = Ri,R = R,n =n,r =r,sd = sd,S_0 = S_0,K = K) for _ in range(N)])

def calculate_Strat_opt_est_vec(N = 10,N1 = 100,m = 10,R = 100,n = 1,r = 0.05,sd = 0.25,S_0 = 100,K = 100):
    ps = np.zeros(m) + 1 / m
    variances = np.zeros(m)
    for i in range(m):
        s =  np.array([generate_and_calculate_I(n=n, r=r, sd=sd, S_0=S_0, i=i + 1, m=m, K=K) for _ in range(N1)])
        variances[i] = np.var(s)
    ss = np.sqrt(variances)
    p = ps * ss
    su = np.sum(p)
    pss = p / su
    Ri = np.ceil(pss * R).astype(int)
    return np.array([calculate_Strat_est(ps = ps,Ri = Ri,R = R,n =n,r =r,sd = sd,S_0 = S_0,K = K) for _ in range(N)])



### control est

def calculate_control_est(R = 100,n = 1,r = 0.05,sd = 0.25,S_0 = 100,K = 100):
    X = np.zeros(R)
    Y = np.zeros(R)
    for i in range(R):
        i_b = generate_and_calculate_I(n =n,r =r,sd = sd,S_0 = S_0,K = K,for_Control_variates= True)
        Y[i] = i_b[0]
        X[i] = i_b[1]
    cov_mat = np.stack((X, Y), axis=0)
    X_var = cov_mat[0][0]
    cov_XY = cov_mat[0][1]
    c = -cov_XY/X_var
    return np.mean(Y)-c*np.mean(X)

















