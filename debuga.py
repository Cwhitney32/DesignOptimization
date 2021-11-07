import numpy as np
import matplotlib.pyplot as plt
from line_search_grg import*
from sk_solve import*

alpha=1
conv=1
error=1

sol=[]
err=[]
itt=[]


def f(x):
    return x[0]**2+x[1]**2+x[2]**2

def h(x):
    return np.array([x[0]**2/4+x[1]**2/5+x[2]**2/25-1, x[0]+x[1]-x[2]])

def pfpd(x):
    return 2*x[2]

def pfps(x):    
    return np.array([2*x[0], 2*x[1]])

def phps(x):
    return np.array([[(1/2)*x[0], 1],[(2/5)*x[1], 1]])

def phpd(x):
    return np.array([(2/25)*x[2], 1])

def dfdd_eqn(x):
    return pfpd(x0)-np.matmul(np.matmul(pfps(x0),np.linalg.inv(phps(x0))),phpd(x0))

k=0

e=10e-3

#state variables x1 x2
sk = np.array([1,2])

#decision variables x3
dk = np.array([3])

#define x argument
x0=np.concatenate([sk,dk])

dfdd=dfdd_eqn(x0)

while conv >= e:

    ak=line_search_grg(f,dfdd_eqn,sk,dk,phps,phpd)
    
    dk=dk-ak*dfdd_eqn(np.concatenate([sk,dk]))
    
    sk= sk + ak * np.transpose(np.matmul(np.linalg.inv(phps(np.concatenate([sk,dk]))),phpd(np.concatenate([sk,dk])))*np.transpose(dfdd_eqn(np.concatenate([sk,dk]))))

    sk=sk_solve(h,dk,sk,e,phps)

    sol.append(float(diff))

    conv=np.linalg.norm(dfdd(sk))
    
error=abs(func(sol)-func(xstar))

