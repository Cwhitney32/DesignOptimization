# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 15:45:29 2021

@author: cwhit
"""

import numpy as np
import matplotlib.pyplot as plt
from line_search_bt import*

x_i=np.array([1,1])
convergence_criteria=.001
conv=1
alpha=1
error=1
err=[]
itt=[]

xstar=[-0.14248091,0.7854586]

def func(x):
    return 5*x[0]**2+12*x[0]*x[1]-8*x[0]+10*x[1]**2-14*x[1]+5

def gradient(x):
   return np.array([10*x[0]+12*x[1]-8,12*x[0]+20*x[1]-14])

Hessian=np.array([[10,12],[12,20]])

while conv > convergence_criteria:
    x_k=x_i
    alpha=line_search_bt(x_i,func,gradient,alpha,.2,.8)
    x_i=x_k-alpha*np.matmul(np.linalg.inv(Hessian),gradient(x_k))
    error=func(x_k)-func(xstar)
    err.append(float(error))
    conv=np.linalg.norm(gradient(x_k))
    print(alpha)
    

x=np.array([-2*x_i[0]-3*x_i[1]+1,x_i[0],x_i[1]])
print(x)

plt.plot(err)
plt.yscale("log")