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
    return np.exp(-r)*(S_0*norm.cdf(d1)-K*np.exp(-r)*norm.cdf(d2))



def generate_and_calculate_I(n = 100,r = 0.05,sd = 0.25,S_0 = 100,i =1,m =1,K = 100,B = None,for_Control_variates = False):
    if B is None:
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
    return np.array([calculate_Strat_est(ps = ps,m=m,Ri = Ri,R = R,n =n,r =r,sd = sd,S_0 = S_0,K = K) for _ in range(N)])

def calculate_Strat_opt_est_vec(N = 10,N1 = 100,m = 10,R = 100000,n = 1,r = 0.05,sd = 0.25,S_0 = 100,K = 100):
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
    return np.array([calculate_Strat_est(ps = ps,Ri = Ri,m=m,R = R,n =n,r =r,sd = sd,S_0 = S_0,K = K) for _ in range(N)])

### Antithetic est

def calculate_ant_est(R = 100,n = 100,r = 0.05,sd = 0.25,S_0 = 100,K = 100):
    R1 = int(R/2)
    v = np.zeros(2*R1)
    for i in range(R1):
        B = np.random.normal(size=1)
        v[2*i] = generate_and_calculate_I(n=n, r=r, sd=sd, B = B,S_0=S_0, K=K)
        v[2*i+1] = generate_and_calculate_I(n=n, r=r, sd=sd, B=-B, S_0=S_0, K=K)
    return np.mean(v)

def calculate_ant_est_vec(N = 10,R = 100,n = 1,r = 0.05,sd = 0.25,S_0 = 100,K = 100):
    return np.array([calculate_ant_est(R = R,n =n,r =r,sd = sd,S_0 = S_0,K = K) for _ in range(N)])

### control est

def calculate_control_est(R = 100,n = 1,r = 0.05,sd = 0.25,S_0 = 100,K = 100):
    X = np.zeros(R)
    Y = np.zeros(R)
    for i in range(R):
        i_b = generate_and_calculate_I(n =n,r =r,sd = sd,S_0 = S_0,K = K,for_Control_variates= True)
        Y[i] = i_b[0]
        X[i] = i_b[1]
    X_var = np.var(X)
    cov_XY = np.cov(X,Y)[0][1]
    c = -cov_XY/X_var
    return np.mean(Y)+c*np.mean(X)


def calculate_control_est_vec(N = 10,R = 100,n = 1,r = 0.05,sd = 0.25,S_0 = 100,K = 100):
    return np.array([calculate_control_est(R = R,n =n,r =r,sd = sd,S_0 = S_0,K = K) for _ in range(N)])

#denerate for n = 1
def generate_a_lot_for_n_one(N = 100,R=100):
    n=1
    df = pd.DataFrame({"value":[],"type":[],"n":[],"R":[]})

    calculators = [calculate_CMC_est_vec,calculate_Strat_prop_est_vec,calculate_Strat_opt_est_vec,
                   calculate_ant_est_vec,calculate_control_est_vec]
    names = ["CMC","Strat_prop","Strat_opt","Antithetic","Control"]
    for a,cal,name in zip(range(5),calculators,names):
        dat = pd.DataFrame({"value":cal(n = n,N = N,R = R),"type":[name for _ in range(N)],"n":[n for _ in range(N)]
                            ,"R":[R for _ in range(N)]})
        df = pd.concat([df,dat])
    return df



def zapisz_wyk1(N = 500,R=500):
    generate_a_lot_for_n_one(N = N,R=R).to_csv("data_n_1.csv", index=False)

def generate_a_lot_for_different_R_for_n_1(Rs = np.arange(100,400,50),N = 100):
    df = pd.DataFrame({"value": [], "type": [], "n": [], "R": []})

    for R in Rs:
        dat  =generate_a_lot_for_n_one(R = R,N = N)
        df = pd.concat([df, dat])
    return df
def zapisz_wyk2(Rs = np.arange(2,50,1),N = 100):
    generate_a_lot_for_different_R_for_n_1(Rs = Rs,N = N).to_csv("data_n_1_different_R.csv", index=False)

def generate_a_lot_for_different_n(n,N=100, R=100):
    df = pd.DataFrame({"value": [], "type": [], "n": [], "R": []})

    calculators = [calculate_CMC_est_vec, calculate_Strat_prop_est_vec, calculate_Strat_opt_est_vec]
    names = ["CMC", "Strat_prop", "Strat_opt"]
    for a, cal, name in zip(range(5), calculators, names):
        dat = pd.DataFrame({"value": cal(n=n, N=N, R=R), "type": [name for _ in range(N)], "n": [n for _ in range(N)]
                               , "R": [R for _ in range(N)]})
        df = pd.concat([df, dat])
    return df

def create_for_different_n_(Ns = np.arange(1,100,5) ,N = 10,R = 10):

    df2 = pd.DataFrame({"value": [], "type": [], "n": [], "R": []})
    for n in Ns:
        dat = generate_a_lot_for_different_n(n=n,R=R,N=N)
        df2 = pd.concat([df2, dat])

    return df2


def zapisz_wyk3(Ns = np.arange(1,200,1) ,N = 200,R = 200):
    create_for_different_n_(Ns = Ns ,N = N,R = R).to_csv("data_n_different.csv", index=False)




