import numpy as np
import math as m
import matplotlib.pyplot as plt
from line_search_grg import*
from sk_solve import*


sol=[]
err=[]
itt=0
conv=1

def f(x):
    return x[0]**2+x[1]**2+x[2]**2

def h(x):
    return np.array( [x[0]**2 /4+x[1]**2 /5+x[2]**2 /25-1, x[0] + x[1] - x[2]])

def pfpd(x):
    return 2*x[2]

def pfps(x):    
    return np.array([2*x[0], 2*x[1]])

def phps(x):
    return np.array([[(1/2)*x[0], (2/5)*x[1]],[1, 1]])

def phpd(x):
    return np.array([(2/25)*x[2], -1])

def dfdd_eqn(x):
    return pfpd(x)-np.matmul(np.matmul(pfps(x),np.linalg.inv(phps(x))),phpd(x))

k=0
e=10e-2

#state variables x1 x2
#sk = np.array([2/29*(-2 + m.sqrt(555)),1])

#decision variables x3
#dk = np.array([1/29*(25 + 2*m.sqrt(555))])

#state variables x1 x2
sk = np.array([-2*m.sqrt(15/17),m.sqrt(5/51)])

#decision variables x3
dk = np.array([-5*m.sqrt(5/51)])

#define x argument
x0=np.concatenate([sk,dk])

check=h(np.concatenate([sk,dk]))

dfdd=dfdd_eqn(x0)

while conv >= e:

    ak=line_search_grg(f,dfdd_eqn,sk,dk,phps,phpd)
    #ak=.2
    dk=dk-ak*dfdd_eqn(np.concatenate([sk,dk]))
    
    sk= sk + ak * np.transpose(np.matmul(np.linalg.inv(phps(np.concatenate([sk,dk]))),phpd(np.concatenate([sk,dk])))*np.transpose(dfdd_eqn(np.concatenate([sk,dk]))))

    sk=sk_solve(h,dk,sk,e,phps)

    conv=np.linalg.norm(dfdd_eqn(np.concatenate([sk,dk])))

    sol.append(conv)
    
    itt=itt+1
    print(itt)
print(np.concatenate([sk,dk]))

plt.plot(sol)